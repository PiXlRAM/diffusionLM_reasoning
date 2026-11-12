"""Step-level deletion: which steps of its computation does each model's answer need?

The removable units are a trace's computation and restatement steps from the checked extraction
(results/steps); setup, plan and connective text always stay. Traces with at most
STEP_EXHAUSTIVE_MAX removable steps get every subset scored, which gives the exact set of minimal
subsets above the 90% threshold; longer traces get a greedy search over steps. Only traces whose
extraction passed every check are used.

Writes results/step_deletion/{model}_{condition}_{split}.jsonl with every scored subset.
"""

import argparse
import itertools
import json
import time

from dlm.cache import PassCache
from dlm.config import CACHE_DIR, MODELS, RESULTS_DIR, STEP_EXHAUSTIVE_MAX, STEPS_DIR
from dlm.generate import load_traces
from dlm.models import Scorer
from dlm.steps import REMOVABLE_ROLES, step_token_spans, token_steps
from scripts.exp1_deletion import CONDITIONS, Runner


def step_units(scorer, reasoning, extraction):
    """Token spans (lo, hi) of the removable steps, and their step indices."""
    enc = scorer.tok(reasoning, add_special_tokens=False, return_offsets_mapping=True)
    if enc.input_ids != scorer.encode_text(reasoning):
        raise ValueError("offset tokenization differs from scoring tokenization")
    spans = [tuple(s) if s else None for s in extraction["spans"]]
    per_step = step_token_spans(token_steps(enc.offset_mapping, spans), len(spans))
    units, index = [], []
    for i, s in enumerate(extraction["steps"]):
        toks = per_step[i]
        if s["role"] in REMOVABLE_ROLES and toks:
            units.append((toks[0], toks[-1] + 1))
            index.append(i)
    return units, index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=None, help="limit the number of questions")
    args = parser.parse_args()

    kind = MODELS[args.model]["kind"]
    if args.condition not in CONDITIONS[kind]:
        raise SystemExit(f"{args.model} supports {CONDITIONS[kind]}")
    with (STEPS_DIR / f"{args.model}_{args.split}.jsonl").open() as f:
        extractions = [r for r in map(json.loads, f) if r["ok"]][: args.n]
    traces = load_traces(args.model, args.split)
    out_dir = RESULTS_DIR / "step_deletion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.condition}_{args.split}.jsonl"
    done = {json.loads(l)["qid"] for l in out_path.open()} if out_path.exists() else set()

    cache = PassCache(CACHE_DIR / f"{args.model}.sqlite")
    scorer = Scorer(args.model, cache=cache)
    todo = [e for e in extractions if e["qid"] not in done]
    print(f"{args.model}/{args.condition}/{args.split}: {len(done)} done, {len(todo)} to go -> {out_path}")
    with out_path.open("a") as out:
        for i, ext in enumerate(todo):
            t = time.time()
            r = traces[ext["qid"]]
            item = scorer.build(r["question"], r["reasoning"], r["gold"])
            units, index = step_units(scorer, r["reasoning"], ext)
            run = Runner(scorer, item, ext["qid"], args.condition)
            run.chunks = units
            base = run.logprob([])
            record = {"qid": ext["qid"], "model": args.model, "condition": args.condition,
                      "estimator": scorer.default_estimator, "base": base, "units": units,
                      "unit_steps": index, "roles": [ext["steps"][s]["role"] for s in index],
                      "final_step": ext["final_step"], "chain": ext["chain"]}
            if not units:
                record["search"] = "none"
                record["subsets"] = []
            elif len(units) <= STEP_EXHAUSTIVE_MAX:
                record["search"] = "exhaustive"
                record["subsets"] = [
                    [list(c), run.logprob(list(c))]
                    for k in range(1, len(units) + 1)
                    for c in itertools.combinations(range(len(units)), k)
                ]
            else:
                record["search"] = "greedy"
                record["greedy_order"], record["greedy"] = run.greedy()
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i + 1}] q{ext['qid']} units={len(units)} {record['search']} base={base:.3f} "
                  f"passes={run.passes} {time.time() - t:.1f}s")
    cache.close()


if __name__ == "__main__":
    main()
