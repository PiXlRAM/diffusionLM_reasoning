"""Summarize Experiment 2 final-line conflict results."""

import argparse
import json
import random
from statistics import mean, median

from dlm.config import RESULTS_DIR


def bootstrap_ci(vals, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    runs = {}
    for path in sorted((RESULTS_DIR / "exp2").glob(f"*_final_conflict_{args.split}.jsonl")):
        model = path.name.split("_final_conflict_")[0]
        runs[model] = [json.loads(l) for l in path.open()]

    if not runs:
        raise SystemExit("no exp2 final_conflict results found")

    print(f"runs: {', '.join(f'{m} ({len(rs)})' for m, rs in sorted(runs.items()))}\n")
    for model, rows in sorted(runs.items()):
        follow = [r["follows_plant"] for r in rows]
        drop = [r["gold_drop"] for r in rows]
        ratio = [r["edit_plant_p"] / r["edit_gold_p"] if r["edit_gold_p"] > 0 else float("inf") for r in rows]
        lo, hi = bootstrap_ci(follow)
        print(f"=== {model} (n={len(rows)}) ===")
        print(f"  follows planted (P_plant > P_gold): {mean(follow):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  median P(gold) baseline / after edit: "
              f"{median(r['base_p'] for r in rows):.3f} / {median(r['edit_gold_p'] for r in rows):.3f}")
        print(f"  median P(planted) after edit: {median(r['edit_plant_p'] for r in rows):.3f}")
        print(f"  median log gold drop: {median(drop):.3f}")
        explicit = [r for r in rows if r.get("equation_explicit")]
        result_only = [r for r in rows if r.get("equation_explicit") is False]
        if explicit:
            print(f"  follow | explicit eq (n={len(explicit)}): {mean(r['follows_plant'] for r in explicit):.3f}")
        if result_only:
            print(f"  follow | result-only (n={len(result_only)}): {mean(r['follows_plant'] for r in result_only):.3f}")
        print()

    if len(runs) >= 2:
        models = sorted(runs)
        by = {m: {r["qid"]: r for r in runs[m]} for m in models}
        shared = set.intersection(*(set(by[m]) for m in models))
        print(f"paired on {len(shared)} shared questions:")
        a, b = models[0], models[1]
        diffs = [float(by[a][q]["follows_plant"]) - float(by[b][q]["follows_plant"]) for q in shared]
        lo, hi = bootstrap_ci(diffs)
        print(f"  follow rate {a} - {b}: {mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
