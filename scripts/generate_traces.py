"""Generate each model's own greedy solutions on a fixed random sample of GSM8K questions.

Appends one JSON line per question to results/traces/{model}_{split}.jsonl and skips questions
already written, so an interrupted job can simply be resubmitted.
"""

import argparse
import json
import random
import time

from dlm.config import MODELS, TRACES_DIR
from dlm.data import load_gsm8k, split_reference
from dlm.generate import generate_trace, trace_path
from dlm.models import Scorer


def sample_qids(n_total, n, seed):
    """First n of a fixed shuffle, so a smaller run is always a subset of a larger one."""
    order = list(range(n_total))
    random.Random(seed).shuffle(order)
    return sorted(order[:n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_gsm8k(args.split)
    qids = sample_qids(len(data), args.n, args.seed)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    path = trace_path(args.model, args.split)
    done = set()
    if path.exists():
        done = {json.loads(line)["qid"] for line in path.open()}
    todo = [q for q in qids if q not in done]
    print(f"{args.model}/{args.split}: {len(done)} done, {len(todo)} to go -> {path}")
    if not todo:
        return

    scorer = Scorer(args.model)
    with path.open("a") as f:
        for i, qid in enumerate(todo, 1):
            t = time.time()
            _, gold = split_reference(data[qid]["answer"])
            record = generate_trace(scorer, data[qid]["question"])
            record.update(
                qid=qid,
                question=data[qid]["question"],
                gold=gold,
                revision=scorer.fingerprint["revision"],
                correct=(not record["truncated"]) and record["pred"] == gold,
            )
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(
                f"[{i}/{len(todo)}] q{qid} pred={record['pred']} gold={gold} "
                f"correct={record['correct']} truncated={record['truncated']} "
                f"tokens={record['n_gen_tokens']} {time.time() - t:.1f}s"
            )
            if record["pred"] is None:
                print(f"    no \\boxed answer; output ends: {record['raw'][-200:]!r}")


if __name__ == "__main__":
    main()
