"""Summarize step-level deletion results.

For each exhaustive run, a question's answer "survives" a removal set if
p / p_base >= 0.9. Reports how small the kept step set can get, whether the
final / chain steps are necessary, and paired model comparisons.
"""

import argparse
import json
import math
import random
from statistics import mean, median

from dlm.config import RESULTS_DIR

THRESHOLD = 0.9


def load(split):
    runs = {}
    for path in sorted((RESULTS_DIR / "step_deletion").glob(f"*_{split}.jsonl")):
        model, condition = path.stem.rsplit("_", 1)[0].split("_", 1)
        runs[(model, condition)] = {r["qid"]: r for r in map(json.loads, path.open())}
    return runs


def summarize(rec):
    base = rec["base"]
    n = len(rec["units"])
    roles = rec["roles"]
    final = rec.get("final_step")
    chain = set(rec.get("chain") or [])
    unit_steps = rec["unit_steps"]  # step index per unit

    def ratio(lp):
        return math.exp(lp - base)

    out = {
        "n_units": n,
        "base_p": math.exp(base),
        "search": rec.get("search"),
    }
    if n == 0 or rec.get("search") == "none":
        out.update(
            kept_min=0, retained=0.0, final_necessary=False, chain_sufficient=False,
            final_only_ok=False, drop_final_ok=False,
        )
        return out

    # surviving removals: sets of removed unit indices with p still high
    surviving = [set()]  # remove nothing always survives
    if rec.get("search") == "exhaustive":
        for removed, lp in rec["subsets"]:
            if ratio(lp) >= THRESHOLD:
                surviving.append(set(removed))
    else:
        # greedy curve: (frac_removed, lp); approximate kept from last point still above threshold
        order = rec.get("greedy_order") or []
        curve = rec.get("greedy") or []
        removed = set()
        best = set()
        for u, (_, lp) in zip(order, curve):
            removed.add(u)
            if ratio(lp) >= THRESHOLD:
                best = set(removed)
            else:
                break
        surviving = [set(), best]

    max_removed = max(len(s) for s in surviving)
    kept_min = n - max_removed
    out["kept_min"] = kept_min
    out["retained"] = kept_min / n

    # unit index of the final computation, if it was removable
    final_units = [i for i, s in enumerate(unit_steps) if s == final]
    chain_units = {i for i, s in enumerate(unit_steps) if s in chain}

    # final necessary: every max-removal surviving set still keeps the final unit
    max_sets = [s for s in surviving if len(s) == max_removed]
    if final_units:
        fu = final_units[0]
        out["final_necessary"] = all(fu not in s for s in max_sets)
        # drop only final
        drop = {fu}
        if rec.get("search") == "exhaustive":
            match = next((lp for rem, lp in rec["subsets"] if set(rem) == drop), None)
            out["drop_final_ok"] = match is not None and ratio(match) >= THRESHOLD
        else:
            out["drop_final_ok"] = None
        # keep only final (= remove all other units)
        keep_only_final = set(range(n)) - {fu}
        if rec.get("search") == "exhaustive":
            match = next((lp for rem, lp in rec["subsets"] if set(rem) == keep_only_final), None)
            out["final_only_ok"] = match is not None and ratio(match) >= THRESHOLD
        else:
            out["final_only_ok"] = None
    else:
        out["final_necessary"] = False
        out["drop_final_ok"] = None
        out["final_only_ok"] = None

    # chain sufficient: can remove every non-chain unit and still survive
    if chain_units and rec.get("search") == "exhaustive":
        remove_non_chain = set(range(n)) - chain_units
        match = next((lp for rem, lp in rec["subsets"] if set(rem) == remove_non_chain), None)
        out["chain_sufficient"] = match is not None and ratio(match) >= THRESHOLD
    else:
        out["chain_sufficient"] = None

    out["n_computation"] = sum(1 for r in roles if r == "computation")
    out["n_restatement"] = sum(1 for r in roles if r == "restatement")
    return out


def bootstrap_ci(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def paired(label, a, b, metric, qids):
    diffs = [a[q][metric] - b[q][metric] for q in qids if a[q][metric] is not None and b[q][metric] is not None]
    if not diffs:
        print(f"  {label:32s} {metric:18s}  (no paired data)")
        return
    lo, hi = bootstrap_ci(diffs)
    print(f"  {label:32s} {metric:18s} mean diff {mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(diffs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    runs = load(args.split)
    if not runs:
        raise SystemExit("no step_deletion results found")
    stats = {key: {q: summarize(r) for q, r in recs.items()} for key, recs in runs.items()}
    complete = set.intersection(*(set(s) for s in stats.values()))
    print(f"runs: {', '.join(f'{m}/{c} ({len(s)})' for (m, c), s in sorted(stats.items()))}")
    print(f"questions in every run: {len(complete)}\n")

    for name, qids in (("all completed in every run", sorted(complete)),):
        if not qids:
            # fall back: per-run summary even if intersection empty
            qids = None
        print(f"=== {name if qids else 'per-run (intersection empty)'} ===")
        print(f"{'run':14s} {'n':>4s} {'kept_min med':>12s} {'retained':>9s} "
              f"{'final nec':>9s} {'final-only':>10s} {'drop final':>10s} {'chain ok':>9s}")
        for (model, condition), s in sorted(stats.items()):
            rows = [s[q] for q in (qids or s)]
            def rate(key):
                vals = [r[key] for r in rows if r[key] is not None]
                return mean(vals) if vals else float("nan")
            print(
                f"{model + '/' + condition:14s} "
                f"{len(rows):4d} "
                f"{median(r['kept_min'] for r in rows):12.1f} "
                f"{median(r['retained'] for r in rows):9.2f} "
                f"{rate('final_necessary'):9.2f} "
                f"{rate('final_only_ok'):10.2f} "
                f"{rate('drop_final_ok'):10.2f} "
                f"{rate('chain_sufficient'):9.2f}"
            )
        if qids:
            print("paired comparisons (A - B) on retained:")
            pairs = [
                (("llada", "mask"), ("llada", "hide"), "LLaDA mask - hide"),
                (("llada", "hide"), ("llada", "delete"), "LLaDA hide - delete"),
                (("qwen", "hide"), ("qwen", "delete"), "Qwen hide - delete"),
                (("llada", "delete"), ("qwen", "delete"), "LLaDA - Qwen (delete)"),
                (("llada", "hide"), ("qwen", "hide"), "LLaDA - Qwen (hide)"),
            ]
            for a, b, label in pairs:
                if a in stats and b in stats:
                    paired(label, stats[a], stats[b], "retained", qids)
                    paired(label, stats[a], stats[b], "final_necessary", qids)
        print()

    print("column guide:")
    print("  kept_min / retained: smallest # / fraction of removable steps that still keep p>=90% of baseline")
    print("  final nec: among those minimal keep-sets, the final computation is always kept")
    print("  final-only: keeping only the final computation (remove other removable steps) still OK")
    print("  drop final: removing only the final computation still OK")
    print("  chain ok: keeping only the answer's dependency chain still OK")


if __name__ == "__main__":
    main()
