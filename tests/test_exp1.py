from dlm.config import LLADA_MASK_ID
from dlm.models import Item
from scripts.exp1_deletion import Runner, apply, chunk_bounds


class FakeScorer:
    """Log-probability is minus the number of visible copies of token 99 missing, so the
    greedy search should remove every chunk without a 99 before the chunk that has one."""

    default_estimator = "fake"

    def score(self, item, meta=None):
        visible = [t for i, t in enumerate(item.context()) if i not in set(item.hidden)]
        return {"logprob": -float(visible.count(99) == 0), "passes": 1, "cached": False}


def make_item(n):
    reasoning = list(range(n))
    reasoning[19] = 99
    return Item(prompt=[1, 2], reasoning=reasoning, prefix=[5], slots=[7])


def test_chunk_bounds_cover_reasoning():
    assert chunk_bounds(20) == [(0, 8), (8, 16), (16, 20)]


def test_apply_conditions():
    item = make_item(20)
    chunks = chunk_bounds(20)
    masked = apply(item, chunks, [1], "mask")
    assert masked.reasoning[8:16] == [LLADA_MASK_ID] * 8 and len(masked.reasoning) == 20
    deleted = apply(item, chunks, [1], "delete")
    assert len(deleted.reasoning) == 12 and deleted.reasoning[8] == 16
    hidden = apply(item, chunks, [0, 2], "hide")
    assert hidden.reasoning == item.reasoning
    assert hidden.hidden == [2 + i for i in list(range(0, 8)) + list(range(16, 20))]


def test_greedy_keeps_the_needed_chunk_longest():
    run = Runner(FakeScorer(), make_item(20), qid=0, condition="delete")
    order, curve = run.greedy()
    assert order[-1] == 2
    assert [lp for _, lp in curve] == [0.0, 0.0, -1.0]
    assert curve[-1][0] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
