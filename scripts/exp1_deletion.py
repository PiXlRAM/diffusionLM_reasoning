"""Experiment 1: how much of its own reasoning does each model's answer need?

The reasoning is split into 8-token chunks. Greedy deletion repeatedly removes the chunk whose
removal hurts the correct answer least, until nothing is left, giving a full curve of answer
probability against fraction removed; the 90%-of-baseline point is read off the curve later.
Controls on the same chunks: truncation from the end and removal in a random order. The end of
every curve (all chunks removed) is the question-only baseline.

Removal conditions: mask (LLaDA only; chunk replaced by mask tokens), delete (chunk's positions
removed), hide (attention mask 0 on the chunk, positions unchanged).
"""

import argparse
import json
import random
import time
from dataclasses import replace

from dlm.cache import PassCache
from dlm.config import CACHE_DIR, LLADA_MASK_ID, MODELS, RESULTS_DIR
from dlm.generate import load_both_correct, load_traces
from dlm.models import Scorer

CHUNK = 8
CONDITIONS = {"diffusion": ("mask", "delete", "hide"), "ar": ("delete", "hide")}


def chunk_bounds(n):
    return [(lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]


def apply(item, chunks, removed, condition):
    spans = [chunks[c] for c in sorted(removed)]
    if condition == "hide":
        for lo, hi in spans:
            item = item.hide_reasoning(lo, hi)
        return item
    gone = {i for lo, hi in spans for i in range(lo, hi)}
    if condition == "mask":
        reasoning = [LLADA_MASK_ID if i in gone else t for i, t in enumerate(item.reasoning)]
    else:
        reasoning = [t for i, t in enumerate(item.reasoning) if i not in gone]
    return replace(item, reasoning=reasoning)


class Runner:
    def __init__(self, scorer, item, qid, condition):
        self.scorer, self.item, self.qid, self.condition = scorer, item, qid, condition
        self.chunks = chunk_bounds(len(item.reasoning))
        self.passes = 0

    def logprob(self, removed):
        meta = {"question": self.qid, "condition": self.condition, "removed": sorted(removed)}
        s = self.scorer.score(apply(self.item, self.chunks, removed, self.condition), meta=meta)
        self.passes += 0 if s["cached"] else s["passes"]
        return s["logprob"]

    def frac(self, removed):
        return sum(self.chunks[c][1] - self.chunks[c][0] for c in removed) / len(self.item.reasoning)

    def ordered_curve(self, order):
        removed, curve = [], []
        for c in order:
            removed.append(c)
            curve.append([self.frac(removed), self.logprob(removed)])
        return curve

    def greedy(self):
        removed, remaining, order, curve = [], list(range(len(self.chunks))), [], []
        while remaining:
            best = max(remaining, key=lambda c: self.logprob(removed + [c]))
            removed.append(best)
            remaining.remove(best)
            order.append(best)
            curve.append([self.frac(removed), self.logprob(removed)])
        return order, curve


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
    qids = load_both_correct(args.split)[: args.n]
    traces = load_traces(args.model, args.split)
    out_dir = RESULTS_DIR / "exp1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.condition}_{args.split}.jsonl"
    done = {json.loads(l)["qid"] for l in out_path.open()} if out_path.exists() else set()

    cache = PassCache(CACHE_DIR / f"{args.model}.sqlite")
    scorer = Scorer(args.model, cache=cache)
    print(f"{args.model}/{args.condition}/{args.split}: {len(done)} done, "
          f"{len(qids) - len(done & set(qids))} to go -> {out_path}")
    with out_path.open("a") as out:
        for i, qid in enumerate(q for q in qids if q not in done):
            t = time.time()
            r = traces[qid]
            item = scorer.build(r["question"], r["reasoning"], r["gold"])
            run = Runner(scorer, item, qid, args.condition)
            base = run.logprob([])
            order, greedy = run.greedy()
            n_chunks = len(run.chunks)
            truncation = run.ordered_curve(list(range(n_chunks - 1, -1, -1)))
            shuffled = list(range(n_chunks))
            random.Random(qid).shuffle(shuffled)
            rand = run.ordered_curve(shuffled)
            record = {
                "qid": qid, "model": args.model, "condition": args.condition,
                "estimator": scorer.default_estimator, "n_tokens": len(item.reasoning),
                "chunks": run.chunks, "base": base, "greedy_order": order, "greedy": greedy,
                "truncation": truncation, "random_order": shuffled, "random": rand,
                "question_only": greedy[-1][1],
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i + 1}] q{qid} chunks={n_chunks} base={base:.3f} "
                  f"question_only={record['question_only']:.3f} passes={run.passes} "
                  f"{time.time() - t:.1f}s")
    cache.close()


if __name__ == "__main__":
    main()
