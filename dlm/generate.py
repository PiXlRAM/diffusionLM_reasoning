import torch

from dlm.cache import make_key
from dlm.config import (
    ANSWER_PREFIX,
    GENERATION,
    LLADA_MASK_ID,
    MODELS,
    PROMPT_TEMPLATE,
    TRACES_DIR,
)
from dlm.data import extract_number, split_at_first_answer, split_generation


def trace_path(model, split):
    """Trace file named by a hash of everything that shapes generation, so configs never mix."""
    fields = {"revision": MODELS[model]["revision"], "prompt": PROMPT_TEMPLATE, "gen": GENERATION[model]}
    if GENERATION[model].get("layout") == "reasoning_as_infilling":
        fields["answer_prefix"] = ANSWER_PREFIX
    tag = make_key(fields)[:8]
    return TRACES_DIR / f"{model}_{split}_{tag}.jsonl"


@torch.inference_mode()
def llada_generate(model, prompt_ids, canvas, block_length, device, answer_start=None):
    """Greedy semi-autoregressive LLaDA decoding of a pre-filled canvas.

    `canvas` holds LLADA_MASK_ID wherever the model should write and fixed token ids elsewhere.
    Blocks are filled left to right; within a block, each step commits the single masked position
    with the lowest predictive entropy, using its most likely token. If `answer_start` is given,
    blocks cover only canvas[:answer_start] and canvas[answer_start:] is decoded as one final
    block, so no answer token is written before the reasoning is complete.
    """
    prompt_len = len(prompt_ids)
    x = torch.tensor([list(prompt_ids) + list(canvas)], dtype=torch.long, device=device)
    split = len(canvas) if answer_start is None else answer_start
    blocks = [(lo, min(lo + block_length, split)) for lo in range(0, split, block_length)]
    if split < len(canvas):
        blocks.append((split, len(canvas)))
    for lo, hi in blocks:
        lo, hi = prompt_len + lo, prompt_len + hi
        while True:
            masked = x[0, lo:hi] == LLADA_MASK_ID
            if not masked.any():
                break
            logits = model(x).logits[0, lo:hi].float()
            logp = torch.log_softmax(logits, dim=-1)
            entropy = -(logp.exp() * logp).sum(dim=-1)
            entropy[~masked] = torch.inf
            i = int(entropy.argmin())
            x[0, lo + i] = logits[i].argmax()
    return x[0, prompt_len:].tolist()


@torch.inference_mode()
def qwen_generate(model, tok, prompt_ids, max_new_tokens, device, **_):
    ids = torch.tensor([prompt_ids], device=device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    return out[0, len(prompt_ids) :].tolist()


def first_stop(ids, stops):
    return next((i for i, t in enumerate(ids) if t in stops), None)


def _llada_trace(scorer, prompt, settings, stops):
    prefix = scorer.encode_text(ANSWER_PREFIX)
    n_answer = settings["answer_slots"]
    n_reason = settings["gen_length"] - len(prefix) - n_answer
    canvas = [LLADA_MASK_ID] * n_reason + prefix + [LLADA_MASK_ID] * n_answer
    answer_start = n_reason if settings["answer_block"] == "after_reasoning" else None
    out = llada_generate(
        scorer.model, prompt, canvas, settings["block_length"], scorer.device, answer_start
    )

    reasoning_ids = out[:n_reason]
    end = first_stop(reasoning_ids, stops)
    reasoning_ids = reasoning_ids if end is None else reasoning_ids[:end]
    answer_ids = out[n_reason + len(prefix) :]
    answer_end = first_stop(answer_ids, stops)
    answer_text = scorer.tok.decode(answer_ids[:answer_end], skip_special_tokens=True)
    reasoning_text = scorer.tok.decode(reasoning_ids, skip_special_tokens=True)
    reasoning, first_pred, answer_sentence, filler = split_at_first_answer(reasoning_text)
    slot_pred = extract_number(answer_text.split("}")[0])
    return {
        "raw": reasoning_text + ANSWER_PREFIX + answer_text,
        "reasoning": reasoning,
        "answer_sentence": answer_sentence,
        "filler": filler,
        "answer_block": answer_text,
        "first_pred": first_pred,
        "slot_pred": slot_pred,
        "pred": first_pred if first_pred is not None else slot_pred,
        "answers_agree": first_pred is None or first_pred == slot_pred,
        "truncated": first_pred is None and end is None,
        "n_gen_tokens": len(reasoning_ids),
    }


def _qwen_trace(scorer, prompt, settings, stops):
    out = qwen_generate(scorer.model, scorer.tok, prompt, device=scorer.device, **settings)
    end = first_stop(out, stops)
    gen_ids = out if end is None else out[:end]
    text = scorer.tok.decode(gen_ids, skip_special_tokens=True)
    reasoning, pred, answer_sentence = split_generation(text)
    return {
        "raw": text,
        "reasoning": reasoning,
        "answer_sentence": answer_sentence,
        "pred": pred,
        "truncated": end is None,
        "n_gen_tokens": len(gen_ids),
    }


def generate_trace(scorer, question):
    """Generate one greedy solution and split it into reasoning and predicted answer."""
    settings = GENERATION[scorer.name]
    prompt = scorer.encode_prompt(question)
    stops = {scorer.end_id, scorer.tok.eos_token_id}
    make = _llada_trace if scorer.kind == "diffusion" else _qwen_trace
    record = make(scorer, prompt, settings, stops)
    record["n_reasoning_tokens"] = len(scorer.encode_text(record["reasoning"]))
    record["generation"] = settings
    return record
