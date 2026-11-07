import torch

from dlm.config import GENERATION, LLADA_MASK_ID
from dlm.data import split_generation


@torch.inference_mode()
def llada_generate(model, prompt_ids, gen_length, steps, block_length, device):
    """Greedy semi-autoregressive LLaDA decoding with low-confidence remasking.

    The canvas is split into blocks filled left to right; within a block, each step commits the
    masked positions the model is most confident about.
    """
    prompt_len = len(prompt_ids)
    x = torch.full((1, prompt_len + gen_length), LLADA_MASK_ID, dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_ids, device=device)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for b in range(num_blocks):
        block_end = prompt_len + (b + 1) * block_length
        n_masked = int((x[0, block_end - block_length : block_end] == LLADA_MASK_ID).sum())
        base, extra = divmod(n_masked, steps_per_block)
        for i in range(steps_per_block):
            k = base + (1 if i < extra else 0)
            if k == 0:
                continue
            masked = x[0] == LLADA_MASK_ID
            logits = model(x).logits[0]
            pred = logits.argmax(dim=-1)
            confidence = torch.softmax(logits.float(), dim=-1).gather(-1, pred[:, None]).squeeze(-1)
            confidence[~masked] = -torch.inf
            confidence[block_end:] = -torch.inf
            chosen = confidence.topk(k).indices
            x[0, chosen] = pred[chosen]

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


def generate_trace(scorer, question):
    """Generate one greedy solution and split it into reasoning and predicted answer."""
    settings = GENERATION[scorer.name]
    prompt = scorer.encode_prompt(question)
    if scorer.kind == "diffusion":
        out = llada_generate(
            scorer.model,
            prompt,
            settings["gen_length"],
            settings["steps"],
            settings["block_length"],
            scorer.device,
        )
    else:
        out = qwen_generate(scorer.model, scorer.tok, prompt, device=scorer.device, **settings)

    stops = {scorer.end_id, scorer.tok.eos_token_id}
    end = next((i for i, t in enumerate(out) if t in stops), None)
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
        "n_reasoning_tokens": len(scorer.encode_text(reasoning)),
        "generation": settings,
    }
