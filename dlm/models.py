import hashlib
import math
from dataclasses import dataclass, field, replace

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from dlm.cache import make_key
from dlm.config import (
    ANSWER_PREFIX,
    ANSWER_SUFFIX,
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
        self.fingerprint = {
            "model": cfg["repo"],
            "revision": cfg["revision"],
            "chat_template": hashlib.sha256(
                (self.tok.chat_template or "").encode()
            ).hexdigest(),
            "prompt_template": PROMPT_TEMPLATE,
            "answer_prefix": ANSWER_PREFIX,
            "answer_suffix": ANSWER_SUFFIX,
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
            slots=self.encode_text(answer) + self.encode_text(ANSWER_SUFFIX) + [self.end_id],
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

    def _slot_logprobs(self, ids, positions, targets, hidden=()):
        """Log-probability of targets[i] at positions[i] from one forward pass."""
        logprobs = torch.log_softmax(self._logits(ids, positions, hidden), dim=-1)
        return logprobs[torch.arange(len(targets)), torch.tensor(targets)].tolist()

    def _slot_ids_and_positions(self, item):
        ctx = item.context()
        start, n = len(ctx), len(item.slots)
        if self.kind == "ar":
            return ctx + item.slots, list(range(start - 1, start + n - 1))
        return ctx + [LLADA_MASK_ID] * n, list(range(start, start + n))

    @torch.inference_mode()
    def slot_alternatives(self, item, k=5):
        """Top-k tokens the model predicts at each answer slot (one-pass view), for diagnostics."""
        ids, positions = self._slot_ids_and_positions(item)
        top = torch.softmax(self._logits(ids, positions, item.hidden), dim=-1).topk(k, dim=-1)
        return [
            [(self.tok.convert_ids_to_tokens(t), p) for t, p in zip(ids_row.tolist(), probs.tolist())]
            for ids_row, probs in zip(top.indices, top.values)
        ]

    def _score_uncached(self, item, estimator):
        if self.kind == "ar" or estimator == "joint":
            ids, positions = self._slot_ids_and_positions(item)
            return self._slot_logprobs(ids, positions, item.slots, item.hidden), 1
        ctx = item.context()
        start, n = len(ctx), len(item.slots)
        per_slot = []
        for i in range(n):
            ids = ctx + item.slots[:i] + [LLADA_MASK_ID] * (n - i)
            per_slot += self._slot_logprobs(ids, [start + i], [item.slots[i]], item.hidden)
        return per_slot, n

    def score(self, item, estimator="joint", meta=None):
        """Probability of the answer slots given the context. Cached when a cache is set."""
        if estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {estimator!r}")
        if self.kind == "ar":
            estimator = "exact"
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
