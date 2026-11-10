"""Pick LLaDA's multi-token answer estimator: one pass ("joint") or slot by slot ("sequential").

On the train both-correct questions, corrupt each LLaDA trace with random contiguous spans,
masked or deleted, and score every corruption both ways. One pass is adopted if the two
estimators rank the probability drops the same way (Spearman >= 0.95) and agree on whether a
corruption crosses the 90%-of-baseline threshold used by greedy deletion (agreement >= 0.95).
"""

import argparse
import json
import random
from dataclasses import replace

from scipy.stats import spearmanr

from dlm.cache import PassCache
from dlm.config import CACHE_DIR, LLADA_MASK_ID, RESULTS_DIR
from dlm.generate import load_both_correct, load_traces
from dlm.models import Scorer

THRESHOLD = 0.9
MIN_SPEARMAN = 0.95
MIN_AGREEMENT = 0.95


def corrupt(item, condition, start, end):
    reasoning = list(item.reasoning)
    if condition == "mask":
        reasoning[start:end] = [LLADA_MASK_ID] * (end - start)
    else:
        del reasoning[start:end]
    return replace(item, reasoning=reasoning)


def random_spans(k, n, rng):
    spans = []
    for _ in range(n):
        length = max(1, round(k * rng.uniform(0.1, 0.5)))
        start = rng.randrange(0, k - length + 1)
        spans.append((start, start + length))
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans", type=int, default=5, help="spans per question, each masked and deleted")
    parser.add_argument("--n", type=int, default=None, help="limit the number of questions")
    args = parser.parse_args()

    qids = load_both_correct("train")[: args.n]
    traces = load_traces("llada", "train")
    cache = PassCache(CACHE_DIR / "llada.sqlite")
    scorer = Scorer("llada", cache=cache)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pilot_estimator.jsonl"
    rows = []

    with out_path.open("w") as out:
        for qid in qids:
            r = traces[qid]
            item = scorer.build(r["question"], r["reasoning"], r["gold"])
            rng = random.Random(qid)
            base = {
                est: scorer.score(item, est, meta={"question": qid, "condition": "full"})
                for est in ("joint", "sequential")
            }
            for start, end in random_spans(len(item.reasoning), args.spans, rng):
                for condition in ("mask", "delete"):
                    meta = {"question": qid, "condition": condition, "removed": [[start, end]]}
                    changed = corrupt(item, condition, start, end)
                    row = {"qid": qid, "condition": condition, "span": [start, end],
                           "n_reasoning": len(item.reasoning)}
                    for est in ("joint", "sequential"):
                        s = scorer.score(changed, est, meta=meta)
                        row[f"{est}_base"] = base[est]["logprob"]
                        row[f"{est}_logprob"] = s["logprob"]
                        row[f"{est}_drop"] = base[est]["logprob"] - s["logprob"]
                        row[f"{est}_below"] = s["prob"] < THRESHOLD * base[est]["prob"]
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
            print(f"q{qid}: base joint={base['joint']['prob']:.3f} seq={base['sequential']['prob']:.3f}")

    cache.close()
    print(f"\n{len(rows)} corruptions on {len(qids)} questions -> {out_path}")
    verdicts = []
    for condition in ("mask", "delete", "all"):
        sub = [r for r in rows if condition == "all" or r["condition"] == condition]
        rho = spearmanr([r["joint_drop"] for r in sub], [r["sequential_drop"] for r in sub]).statistic
        agree = sum(r["joint_below"] == r["sequential_below"] for r in sub) / len(sub)
        print(f"{condition:6s} n={len(sub)} spearman={rho:.3f} threshold_agreement={agree:.3f}")
        verdicts.append(rho >= MIN_SPEARMAN and agree >= MIN_AGREEMENT)
    print("verdict:", "use one pass (joint)" if all(verdicts) else "use sequential")


if __name__ == "__main__":
    main()
