"""Summarize Experiment 1 deletion curves.

For every model and removal condition, per question:
  - retained: smallest fraction of the reasoning that must stay before the correct answer's
    probability first falls below 90% of its full-reasoning baseline (greedy, truncation, random);
  - auc: area under the curve of p / p_baseline (capped at 1) against fraction removed;
  - question_only: p with all reasoning removed, relative to baseline;
  - final_chunk_last: whether greedy removed the reasoning's last chunk last.
Questions enter the headline subset when every model's baseline is at least 0.5.
Paired comparisons use the same questions, with bootstrap 95% confidence intervals.
Writes mean curves to results/exp1/mean_curves.csv for plotting.
"""

import argparse
import json
import math
import random
from statistics import mean, median

from dlm.config import RESULTS_DIR

THRESHOLD = 0.9
MIN_BASE = 0.5
GRID = [i / 20 for i in range(21)]


def load(split):
    runs = {}
    for path in sorted((RESULTS_DIR / "exp1").glob(f"*_{split}.jsonl")):
        model, condition = path.stem.rsplit("_", 1)[0].split("_", 1)
        runs[(model, condition)] = {r["qid"]: r for r in map(json.loads, path.open())}
    return runs


def ratio_curve(base, curve):
    """(fraction removed, p / p_base capped at 1), starting from (0, 1)."""
    return [(0.0, 1.0)] + [(f, min(1.0, math.exp(lp - base))) for f, lp in curve]


def retained(points):
    removable = 0.0
    for f, r in points:
        if r < THRESHOLD:
            break
        removable = f
    return 1.0 - removable


def auc(points):
    return sum((f2 - f1) * (r1 + r2) / 2 for (f1, r1), (f2, r2) in zip(points, points[1:]))


def at(points, x):
    for (f1, r1), (f2, r2) in zip(points, points[1:]):
        if f1 <= x <= f2:
            return r1 if f2 == f1 else r1 + (r2 - r1) * (x - f1) / (f2 - f1)
    return points[-1][1]


def summarize(record):
    out = {}
    for name in ("greedy", "truncation", "random"):
        pts = ratio_curve(record["base"], record[name])
        out[f"{name}_retained"] = retained(pts)
        out[f"{name}_auc"] = auc(pts)
        out[f"{name}_curve"] = [at(pts, x) for x in GRID]
    out["question_only"] = min(1.0, math.exp(record["question_only"] - record["base"]))
    order = record["greedy_order"]
    out["final_chunk_last"] = order[-1] == len(record["chunks"]) - 1
    return out


def bootstrap_ci(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def paired(label, a, b, metric, qids):
    diffs = [a[q][metric] - b[q][metric] for q in qids]
    lo, hi = bootstrap_ci(diffs)
    print(f"  {label:32s} {metric:18s} mean diff {mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(diffs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    runs = load(args.split)
    if not runs:
        raise SystemExit("no Experiment 1 results found")
    stats = {key: {q: summarize(r) for q, r in recs.items()} for key, recs in runs.items()}
    base = {}
    for (model, _), recs in runs.items():
        for q, r in recs.items():
            base.setdefault(model, {})[q] = math.exp(r["base"])
    models = sorted(base)
    complete = set.intersection(*(set(s) for s in stats.values()))
    high = {q for q in complete if all(base[m].get(q, 0) >= MIN_BASE for m in models)}
    print(f"runs: {', '.join(f'{m}/{c} ({len(s)})' for (m, c), s in sorted(stats.items()))}")
    print(f"questions in every run: {len(complete)}; baseline >= {MIN_BASE} for all models: {len(high)}\n")

    for subset_name, qids in (("high-baseline subset", sorted(high)), ("all questions", sorted(complete))):
        if not qids:
            continue
        print(f"=== {subset_name} (n={len(qids)}) ===")
        print(f"{'run':14s} {'retained@90% greedy':>20s} {'trunc':>7s} {'random':>7s} "
              f"{'AUC greedy':>11s} {'trunc':>7s} {'random':>7s} {'q-only':>7s} {'final last':>10s}")
        for (model, condition), s in sorted(stats.items()):
            rows = [s[q] for q in qids]
            print(
                f"{model + '/' + condition:14s} "
                f"{median(r['greedy_retained'] for r in rows):>20.2f} "
                f"{median(r['truncation_retained'] for r in rows):>7.2f} "
                f"{median(r['random_retained'] for r in rows):>7.2f} "
                f"{mean(r['greedy_auc'] for r in rows):>11.3f} "
                f"{mean(r['truncation_auc'] for r in rows):>7.3f} "
                f"{mean(r['random_auc'] for r in rows):>7.3f} "
                f"{median(r['question_only'] for r in rows):>7.3f} "
                f"{mean(r['final_chunk_last'] for r in rows):>10.2f}"
            )
        print("paired comparisons (A - B):")
        pairs = [
            (("llada", "mask"), ("llada", "hide"), "LLaDA mask - hide"),
            (("llada", "hide"), ("llada", "delete"), "LLaDA hide - delete"),
            (("qwen", "hide"), ("qwen", "delete"), "Qwen hide - delete"),
            (("llada", "delete"), ("qwen", "delete"), "LLaDA - Qwen (delete)"),
            (("llada", "hide"), ("qwen", "hide"), "LLaDA - Qwen (hide)"),
        ]
        for a, b, label in pairs:
            if a in stats and b in stats:
                for metric in ("greedy_retained", "greedy_auc"):
                    paired(label, stats[a], stats[b], metric, qids)
        print()

    out = RESULTS_DIR / "exp1" / "mean_curves.csv"
    with out.open("w") as f:
        f.write("run,order,fraction_removed,mean_ratio,subset\n")
        for subset_name, qids in (("high", sorted(high)), ("all", sorted(complete))):
            for (model, condition), s in sorted(stats.items()):
                for name in ("greedy", "truncation", "random"):
                    for i, x in enumerate(GRID):
                        y = mean(s[q][f"{name}_curve"][i] for q in qids) if qids else float("nan")
                        f.write(f"{model}/{condition},{name},{x:.2f},{y:.4f},{subset_name}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
