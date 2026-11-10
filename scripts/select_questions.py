"""Summarize generated traces and write the set of questions every model answers correctly."""

import argparse
import json
from statistics import mean

from dlm.config import MODELS, TRACES_DIR
from dlm.generate import trace_path


def load(model, split):
    path = trace_path(model, split)
    print(f"reading {path}")
    return {r["qid"]: r for r in map(json.loads, path.open())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    traces = {m: load(m, args.split) for m in args.models}
    shared = set.intersection(*(set(t) for t in traces.values()))
    correct = {m: {q for q in shared if t[q]["correct"]} for m, t in traces.items()}

    for m, t in traces.items():
        rows = [t[q] for q in shared]
        kept = [r for r in rows if r["correct"]]
        print(
            f"{m}: n={len(rows)} accuracy={len(kept) / len(rows):.3f} "
            f"truncated={sum(r['truncated'] for r in rows)} "
            f"no_answer={sum(r['pred'] is None for r in rows)} "
            f"mean_reasoning_tokens={mean(r['n_reasoning_tokens'] for r in kept) if kept else 0:.0f}"
        )
        if rows and "filler" in rows[0]:
            print(
                f"  answers_disagree={sum(not r['answers_agree'] for r in rows)} "
                f"with_filler={sum(bool(r['filler']) for r in rows)} "
                f"no_answer_in_reasoning={sum(r['first_pred'] is None for r in rows)} "
                f"cut_off_but_slots_correct="
                f"{sum(r['truncated'] and r['slot_pred'] == r['gold'] for r in rows)}"
            )

    both = sorted(set.intersection(*correct.values()))
    print(f"correct for all of {args.models}: {len(both)} / {len(shared)}")
    for m in args.models:
        others = set.union(*(correct[o] for o in args.models if o != m)) if len(args.models) > 1 else set()
        print(f"  only {m}: {len(correct[m] - others)}")

    out = TRACES_DIR / f"both_correct_{args.split}.json"
    out.write_text(json.dumps({"models": args.models, "qids": both}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
