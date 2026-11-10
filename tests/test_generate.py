from types import SimpleNamespace

import torch

from dlm.config import LLADA_MASK_ID
from dlm.generate import llada_generate

VOCAB = 50
FIXED = 7


class FakeModel:
    """Predicts token (position % 10) + 20 everywhere, most sharply at the lowest positions."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        n = x.shape[1]
        logits = torch.zeros(1, n, VOCAB)
        for pos in range(n):
            logits[0, pos, pos % 10 + 20] = 10.0 - 0.01 * pos
        return SimpleNamespace(logits=logits)


def test_fills_every_mask_and_keeps_fixed_tokens():
    prompt = [1, 2, 3]
    canvas = [LLADA_MASK_ID] * 5 + [FIXED, FIXED] + [LLADA_MASK_ID] * 3
    model = FakeModel()
    out = llada_generate(model, prompt, canvas, block_length=4, device="cpu")
    assert len(out) == len(canvas)
    assert LLADA_MASK_ID not in out
    assert out[5:7] == [FIXED, FIXED]
    for i, token in enumerate(out):
        if canvas[i] == LLADA_MASK_ID:
            assert token == (len(prompt) + i) % 10 + 20


def test_one_forward_pass_per_masked_token():
    canvas = [LLADA_MASK_ID] * 6 + [FIXED] + [LLADA_MASK_ID] * 2
    model = FakeModel()
    llada_generate(model, [1], canvas, block_length=4, device="cpu")
    assert model.calls == 8


class LateFirstModel(FakeModel):
    """Most confident at the latest positions, and records every input it sees."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x[0].clone())
        out = super().__call__(x)
        n = x.shape[1]
        for pos in range(n):
            out.logits[0, pos, pos % 10 + 20] = 1.0 + 0.1 * pos
        return out


def test_answer_block_waits_for_reasoning():
    prompt = [1]
    n_reason, n_answer = 6, 3
    canvas = [LLADA_MASK_ID] * n_reason + [FIXED] + [LLADA_MASK_ID] * n_answer
    model = LateFirstModel()
    out = llada_generate(model, prompt, canvas, block_length=4, device="cpu", answer_start=n_reason)
    assert LLADA_MASK_ID not in out
    reason = slice(len(prompt), len(prompt) + n_reason)
    answer = slice(len(prompt) + n_reason + 1, None)
    for x in model.inputs:
        if (x[answer] != LLADA_MASK_ID).any():
            assert not (x[reason] == LLADA_MASK_ID).any()


def test_without_answer_start_answer_can_come_first():
    canvas = [LLADA_MASK_ID] * 6 + [FIXED] + [LLADA_MASK_ID] * 3
    model = LateFirstModel()
    llada_generate(model, [1], canvas, block_length=4, device="cpu")
    answer_first = any(
        (x[8:] != LLADA_MASK_ID).any() and (x[1:7] == LLADA_MASK_ID).any() for x in model.inputs
    )
    assert answer_first


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
