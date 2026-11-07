"""Step-2 checks on GPU: models load, scores are sane, scoring is deterministic, cache works.

Uses GSM8K train questions with their reference solutions, since model-written traces come later.
"""

import argparse
import math
import time

import torch

from dlm.cache import PassCache
from dlm.config import CACHE_DIR, MODELS
from dlm.data import load_gsm8k, split_reference
from dlm.models import Scorer

N_QUESTIONS = 3


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message)
    return condition


def run_model(name, examples, cache):
    print(f"\n===== {name} =====")
    t0 = time.time()
    scorer = Scorer(name, cache=cache)
    print(f"loaded in {time.time() - t0:.1f}s, end marker id {scorer.end_id}")
    estimators = ["joint", "sequential"] if scorer.kind == "diffusion" else ["joint"]
    ok = True

    for qid, (question, reasoning, answer) in examples:
        wrong = str(int(answer) + 1) if answer.lstrip("-").isdigit() else answer + "0"
        item = scorer.build(question, reasoning, answer)
        wrong_item = scorer.build(question, reasoning, wrong)
        slot_tokens = scorer.tok.convert_ids_to_tokens(item.slots)
        print(f"\n[q{qid}] answer={answer} slots={slot_tokens}")
        for token, options in zip(slot_tokens, scorer.slot_alternatives(item)):
            shown = ", ".join(f"{t!r}:{p:.3f}" for t, p in options)
            print(f"    slot {token!r:14s} top5: {shown}")

        for est in estimators:
            scorer.cache = None
            torch.cuda.synchronize()
            t = time.time()
            first = scorer.score(item, est)
            torch.cuda.synchronize()
            per_pass = (time.time() - t) / first["passes"]
            second = scorer.score(item, est)
            bad = scorer.score(wrong_item, est)
            per_slot = ", ".join(f"{math.exp(lp):.3f}" for lp in first["per_slot"])
            print(
                f"  {est:10s} p(correct)={first['prob']:.4f} p({wrong})={bad['prob']:.4f} "
                f"passes={first['passes']} sec/pass={per_pass:.3f} per-slot=[{per_slot}]"
            )
            ok &= check(0.0 < first["prob"] <= 1.0, f"{est}: probability in (0, 1]")
            ok &= check(first["prob"] > bad["prob"], f"{est}: correct answer beats {wrong}")
            ok &= check(
                first["logprob"] == second["logprob"], f"{est}: identical score on rerun"
            )

            scorer.cache = cache
            meta = {"question": qid, "condition": "reference", "removed": []}
            stored = scorer.score(item, est, meta=meta)
            loaded = scorer.score(item, est, meta=meta)
            ok &= check(
                loaded["cached"] and loaded["logprob"] == stored["logprob"] == first["logprob"],
                f"{est}: cache hit returns the same score",
            )

    scorer.unload()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    data = load_gsm8k("train")
    examples = []
    for qid in range(N_QUESTIONS):
        reasoning, answer = split_reference(data[qid]["answer"])
        examples.append((qid, (data[qid]["question"], reasoning, answer)))

    cache = PassCache(CACHE_DIR / "smoke.sqlite")
    results = {name: run_model(name, examples, cache) for name in args.models}
    cache.close()

    print("\n===== summary =====")
    for name, ok in results.items():
        print(f"{name}: {'ALL PASS' if ok else 'SOME FAILED'}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
