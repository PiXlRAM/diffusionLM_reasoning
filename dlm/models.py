import hashlib
import math
from dataclasses import dataclass, field, replace

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from dlm.cache import make_key
from dlm.config import (
    ANSWER_CLOSINGS,
    ANSWER_PREFIX,
    ESTIMATORS,
    GENERATION,
    LLADA_MASK_ID,
    MODELS,
    PROMPT_TEMPLATE,
)


@dataclass
class Item:
    """Token ids for one scoring input: prompt, reasoning, answer prefix, answer slots."""

    prompt: list
    reasoning: list
    prefix: list
    slots: list
    hidden: list = field(default_factory=list)  # context positions no other token may attend to

    def context(self):
        return self.prompt + self.reasoning + self.prefix

    def hide_reasoning(self, start, end):
        """Copy of this item with reasoning[start:end] hidden; positions stay unchanged."""
        offset = len(self.prompt)
        return replace(self, hidden=sorted(set(self.hidden) | set(range(offset + start, offset + end))))


class Scorer:
    def __init__(self, name, device="cuda", cache=None):
        cfg = MODELS[name]
        self.name = name
        self.kind = cfg["kind"]
        self.device = device
        self.cache = cache
        self.tok = AutoTokenizer.from_pretrained(
            cfg["repo"], revision=cfg["revision"], trust_remote_code=True
        )
        model_cls = AutoModel if self.kind == "diffusion" else AutoModelForCausalLM
        self.model = model_cls.from_pretrained(
            cfg["repo"],
            revision=cfg["revision"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device,
        ).eval()
        self.end_id = self.tok.convert_tokens_to_ids(cfg["end_marker"])
        if self.end_id is None or self.end_id == self.tok.unk_token_id:
            raise ValueError(f"end marker {cfg['end_marker']!r} not in {name} vocabulary")
        self.closing_ids = []
        for text in ANSWER_CLOSINGS:
            ids = self.encode_text(text)
            if len(ids) != 1:
                raise ValueError(f"closing {text!r} is {len(ids)} tokens for {name}")
            self.closing_ids += ids
        self.default_estimator = cfg["estimator"]
        self.fingerprint = {
            "model": cfg["repo"],
            "revision": cfg["revision"],
            "chat_template": hashlib.sha256(
                (self.tok.chat_template or "").encode()
            ).hexdigest(),
            "prompt_template": PROMPT_TEMPLATE,
            "answer_prefix": ANSWER_PREFIX,
            "answer_closings": list(ANSWER_CLOSINGS),
            "end_marker": cfg["end_marker"],
            "dtype": "bfloat16",
            "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        }
        self.fingerprint["generation"] = GENERATION[name]
        if self.kind == "diffusion":
            self.fingerprint["mask_id"] = LLADA_MASK_ID

    def encode_text(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def encode_prompt(self, question):
        messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    def build(self, question, reasoning, answer):
        return Item(
            prompt=self.encode_prompt(question),
            reasoning=self.encode_text(reasoning),
            prefix=self.encode_text(ANSWER_PREFIX),
            slots=self.encode_text(answer) + [self.closing_ids[0], self.end_id],
        )

    @torch.inference_mode()
    def _logits(self, ids, positions, hidden=()):
        """Logits at `positions` from one forward pass, with `hidden` positions masked as keys.

        Hiding uses the attention mask only; every token keeps its original position. Qwen gets
        explicit position_ids so its positions are not recomputed from the mask.
        """
        input_ids = torch.tensor([ids], device=self.device)
        kwargs = {}
        if hidden:
            mask = torch.ones_like(input_ids)
            mask[0, list(hidden)] = 0
            kwargs["attention_mask"] = mask
            if self.kind == "ar":
                kwargs["position_ids"] = torch.arange(len(ids), device=self.device)[None]
        return self.model(input_ids, **kwargs).logits[0, positions].float()

    def _slot_ids_and_positions(self, item):
        ctx = item.context()
        start, n = len(ctx), len(item.slots)
        if self.kind == "ar":
            return ctx + item.slots, list(range(start - 1, start + n - 1))
        return ctx + [LLADA_MASK_ID] * n, list(range(start, start + n))

    @torch.inference_mode()
    def slot_alternatives(self, item, k=5, sequential=False):
        """Top-k tokens the model predicts at each answer slot, for diagnostics.

        By default one pass with every slot masked (LLaDA) or teacher-forced (Qwen). With
        `sequential`, LLaDA's slot i is predicted with the correct slots before it revealed.
        """
        if sequential and self.kind == "diffusion":
            ctx, n = item.context(), len(item.slots)
            logits = torch.cat([
                self._logits(ctx + item.slots[:i] + [LLADA_MASK_ID] * (n - i), [len(ctx) + i], item.hidden)
                for i in range(n)
            ])
        else:
            ids, positions = self._slot_ids_and_positions(item)
            logits = self._logits(ids, positions, item.hidden)
        top = torch.softmax(logits, dim=-1).topk(k, dim=-1)
        return [
            [(self.tok.convert_ids_to_tokens(t), p) for t, p in zip(ids_row.tolist(), probs.tolist())]
            for ids_row, probs in zip(top.indices, top.values)
        ]

    def _logprob_rows(self, ids, positions, hidden):
        return torch.log_softmax(self._logits(ids, positions, hidden), dim=-1)

    def _score_uncached(self, item, estimator):
        """Per-slot log-probabilities: one per answer token, then one for closing plus end marker.

        The closing slot is marginalized over ANSWER_CLOSINGS, so whether the model would close
        with "}" or "}." does not count against the answer.
        """
        ctx, slots, hidden = item.context(), item.slots, item.hidden
        start, digits, end = len(ctx), item.slots[:-2], item.slots[-1]
        d, n = len(digits), len(item.slots)
        if self.kind == "ar":
            closing, digit_lps = [], None
            for c in self.closing_ids:
                rows = self._logprob_rows(ctx + digits + [c, end], list(range(start - 1, start + n - 1)), hidden)
                digit_lps = [rows[i, t].item() for i, t in enumerate(digits)]
                closing.append(rows[d, c] + rows[d + 1, end])
            return digit_lps + [torch.logsumexp(torch.stack(closing), 0).item()], len(self.closing_ids)
        if estimator == "joint":
            rows = self._logprob_rows(ctx + [LLADA_MASK_ID] * n, list(range(start, start + n)), hidden)
            close = torch.logsumexp(rows[d, self.closing_ids], 0) + rows[d + 1, end]
            return [rows[i, t].item() for i, t in enumerate(digits)] + [close.item()], 1
        per_slot = []
        for i in range(d):
            ids = ctx + digits[:i] + [LLADA_MASK_ID] * (n - i)
            per_slot.append(self._logprob_rows(ids, [start + i], hidden)[0, digits[i]].item())
        close_lps = self._logprob_rows(ctx + digits + [LLADA_MASK_ID] * 2, [start + d], hidden)[0]
        closing = [
            close_lps[c] + self._logprob_rows(ctx + digits + [c, LLADA_MASK_ID], [start + d + 1], hidden)[0, end]
            for c in self.closing_ids
        ]
        per_slot.append(torch.logsumexp(torch.stack(closing), 0).item())
        return per_slot, d + 1 + len(self.closing_ids)

    def score(self, item, estimator=None, meta=None):
        """Probability of the answer slots given the context. Cached when a cache is set."""
        if self.kind == "ar" or estimator is None:
            estimator = self.default_estimator
        if self.kind == "diffusion" and estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {estimator!r}")
        key_fields = {
            **self.fingerprint,
            "estimator": estimator,
            "input_ids": item.context(),
            "slots": item.slots,
            "meta": meta or {},
        }
        if item.hidden:
            key_fields["hidden"] = item.hidden
        key = make_key(key_fields)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return {**hit, "cached": True}
        per_slot, passes = self._score_uncached(item, estimator)
        total = sum(per_slot)
        result = {
            "logprob": total,
            "prob": math.exp(total),
            "per_slot": per_slot,
            "passes": passes,
            "estimator": estimator,
        }
        if self.cache is not None:
            self.cache.put(key, {"model": self.name, **(meta or {})}, result)
        return {**result, "cached": False}

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
