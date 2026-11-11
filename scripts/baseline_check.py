"""Validate the scoring layout on each model's own traces before any deletion.

For every both-correct question, score the full reasoning with the appended answer prefix and
check that the correct answer gets high probability and is the top prediction at every slot.
A low or off-top baseline means the appended layout is off-distribution for that model.
"""

import argparse
from statistics import median

from dlm.cache import PassCache
from dlm.config import CACHE_DIR, MODELS
from dlm.generate import load_both_correct, load_traces
from dlm.models import Scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    for name in args.models:
        cache = PassCache(CACHE_DIR / f"{name}.sqlite")
        scorer = Scorer(name, cache=cache)
        for split in args.splits:
            traces = load_traces(name, split)
            probs, top, low = [], 0, []
            for qid in load_both_correct(split):
                r = traces[qid]
                item = scorer.build(r["question"], r["reasoning"], r["gold"])
                s = scorer.score(item, meta={"question": qid, "condition": "full"})
                best = [options[0][0] for options in scorer.slot_alternatives(item, k=1)]
                is_top = best == scorer.tok.convert_ids_to_tokens(item.slots)
                probs.append(s["prob"])
                top += is_top
                if s["prob"] < 0.5 or not is_top:
                    low.append((qid, round(s["prob"], 3), best))
            n = len(probs)
            print(
                f"{name}/{split}: n={n} median_p={median(probs):.3f} "
                f"p>0.5={sum(p > 0.5 for p in probs)}/{n} p>0.9={sum(p > 0.9 for p in probs)}/{n} "
                f"top_at_every_slot={top}/{n}"
            )
            for qid, p, best in low[:10]:
                print(f"    q{qid} p={p} top={best}")
        scorer.unload()
        cache.close()


if __name__ == "__main__":
    main()
