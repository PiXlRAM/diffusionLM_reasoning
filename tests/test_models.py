import math
from types import SimpleNamespace

import torch

from dlm.config import LLADA_MASK_ID
from dlm.models import Item, Scorer

VOCAB = 10
CLOSE_A, CLOSE_B, END = 7, 8, 9


class TableModel:
    """Next-token distribution depends only on the token at each position (Qwen-style)."""

    def __init__(self, table):
        self.table = table

    def __call__(self, input_ids, **_):
        rows = [self.table.get(t, torch.zeros(VOCAB)) for t in input_ids[0].tolist()]
        return SimpleNamespace(logits=torch.stack(rows)[None])


def dist(**probs):
    p = torch.full((VOCAB,), 1e-9)
    for k, v in probs.items():
        p[int(k[1:])] = v
    return torch.log(p)


def fake_scorer(kind, table):
    s = Scorer.__new__(Scorer)
    s.kind, s.device, s.model, s.closing_ids = kind, "cpu", TableModel(table), [CLOSE_A, CLOSE_B]
    return s


def test_ar_closing_is_marginalized():
    table = {
        5: dist(t3=1.0),                      # after the prefix: the answer digit 3
        3: dist(t7=0.6, t8=0.4),              # after the digit: either closing
        CLOSE_A: dist(t9=0.5, t1=0.5),        # end marker after "}"
        CLOSE_B: dist(t9=1.0),                # end marker after "}."
    }
    item = Item(prompt=[1], reasoning=[2], prefix=[5], slots=[3, CLOSE_A, END])
    per_slot, passes = fake_scorer("ar", table)._score_uncached(item, "exact")
    assert passes == 2
    assert math.isclose(per_slot[0], 0.0, abs_tol=1e-6)
    assert math.isclose(math.exp(per_slot[1]), 0.6 * 0.5 + 0.4 * 1.0, rel_tol=1e-5)


def test_diffusion_sequential_closing_is_marginalized():
    item = Item(prompt=[1], reasoning=[2], prefix=[5], slots=[3, CLOSE_A, END])
    s = fake_scorer("diffusion", {})
    s.model = lambda input_ids, **_: SimpleNamespace(
        logits=torch.stack([dist(t3=0.5, t7=0.3, t8=0.2, t9=0.0) if t == LLADA_MASK_ID and i < 5
                            else dist(t9=0.9, t3=0.1) for i, t in enumerate(input_ids[0].tolist())])[None]
    )
    per_slot, passes = s._score_uncached(item, "sequential")
    assert passes == 1 + 1 + 2
    assert math.isclose(math.exp(per_slot[0]), 0.5, rel_tol=1e-5)
    assert math.isclose(math.exp(per_slot[1]), (0.3 + 0.2) * 0.9, rel_tol=1e-5)



def test_hide_reasoning_offsets_by_prompt_and_keeps_tokens():
    item = Item(prompt=[1, 2], reasoning=[10, 11, 12, 13], prefix=[5], slots=[7, 8])
    hidden = item.hide_reasoning(1, 3)
    assert hidden.hidden == [3, 4]
    assert hidden.context() == item.context()
    assert item.hidden == []


def test_hide_reasoning_accumulates_spans():
    item = Item(prompt=[1], reasoning=[10, 11, 12, 13], prefix=[], slots=[7])
    assert item.hide_reasoning(0, 1).hide_reasoning(3, 4).hidden == [1, 4]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
