"""Experiment 2 (copy vs compute): plant a wrong result in the final computation step.

Keeps the equation's left-hand side; changes only the stated result in that step's quote.
Scores the gold answer and the planted answer on the edited reasoning. Following the plant
(P_plant > P_gold) is evidence of copying the last number; keeping gold is evidence of
computing from the earlier chain / question.

Only uses checked extractions with a final computation that produces the gold answer.
Writes results/exp2/{model}_final_conflict_{split}.jsonl
"""

import argparse
import json
import math
import time

from dlm.cache import PassCache
from dlm.config import CACHE_DIR, MODELS, RESULTS_DIR, STEPS_DIR
from dlm.edits import apply_final_result_plant, plant_value
from dlm.generate import load_traces
from dlm.models import Scorer


def usable(ext):
    if not ext.get("ok"):
        return False
    if "answer not in reasoning" in ext.get("flags", []):
        return False
    final = ext.get("final_step")
    if final is None:
        return False
    step = ext["steps"][final]
    if step["role"] != "computation" or step.get("result") is None:
        return False
    span = ext["spans"][final]
    return span is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    with (STEPS_DIR / f"{args.model}_{args.split}.jsonl").open() as f:
        extractions = [r for r in map(json.loads, f) if usable(r)][: args.n]
    traces = load_traces(args.model, args.split)
    out_dir = RESULTS_DIR / "exp2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_final_conflict_{args.split}.jsonl"
    done = {json.loads(l)["qid"] for l in out_path.open()} if out_path.exists() else set()

    cache = PassCache(CACHE_DIR / f"{args.model}.sqlite")
    scorer = Scorer(args.model, cache=cache)
    todo = [e for e in extractions if e["qid"] not in done]
    print(f"{args.model}/final_conflict/{args.split}: {len(done)} done, {len(todo)} to go -> {out_path}")

    with out_path.open("a") as out:
        for i, ext in enumerate(todo):
            t0 = time.time()
            tr = traces[ext["qid"]]
            final = ext["final_step"]
            step = ext["steps"][final]
            span = tuple(ext["spans"][final])
            planted = plant_value(step["result"], tr["gold"])
            edited = apply_final_result_plant(tr["reasoning"], span, step["result"], planted)
            if edited is None:
                print(f"[{i + 1}] q{ext['qid']} SKIP: result {step['result']!r} not found in quote")
                continue
            new_reasoning, a, b = edited

            base_item = scorer.build(tr["question"], tr["reasoning"], tr["gold"])
            edit_gold = scorer.build(tr["question"], new_reasoning, tr["gold"])
            edit_plant = scorer.build(tr["question"], new_reasoning, planted)

            meta = {"question": ext["qid"], "exp": "final_conflict"}
            s_base = scorer.score(base_item, meta={**meta, "cond": "base"})
            s_gold = scorer.score(edit_gold, meta={**meta, "cond": "edit_gold", "planted": planted})
            s_plant = scorer.score(edit_plant, meta={**meta, "cond": "edit_plant", "planted": planted})

            # Distance: tokens from end of planted span to start of answer prefix.
            # Approximate with character offset from plant to end of reasoning.
            plant_tok_start = len(scorer.encode_text(new_reasoning[:a]))
            n_reason = len(scorer.encode_text(new_reasoning))
            dist = n_reason - plant_tok_start

            record = {
                "qid": ext["qid"],
                "model": args.model,
                "gold": tr["gold"],
                "planted": planted,
                "final_step": final,
                "chain": ext["chain"],
                "equation_explicit": step.get("equation_explicit"),
                "expression": step.get("expression"),
                "old_result": step["result"],
                "quote": step["quote"],
                "edit_start": a,
                "edit_end": b,
                "distance_tokens": dist,
                "base_logprob": s_base["logprob"],
                "edit_gold_logprob": s_gold["logprob"],
                "edit_plant_logprob": s_plant["logprob"],
                "base_p": s_base["prob"],
                "edit_gold_p": s_gold["prob"],
                "edit_plant_p": s_plant["prob"],
                "follows_plant": s_plant["logprob"] > s_gold["logprob"],
                "gold_drop": s_base["logprob"] - s_gold["logprob"],
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(
                f"[{i + 1}] q{ext['qid']} {step['result']}->{planted} "
                f"base={s_base['prob']:.3f} gold'={s_gold['prob']:.3f} plant={s_plant['prob']:.3f} "
                f"follow={record['follows_plant']} {time.time() - t0:.1f}s"
            )
    cache.close()


if __name__ == "__main__":
    main()
