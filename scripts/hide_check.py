"""Week-1 hide validation: hiding a span must make its contents invisible to the answer slots.

For each train both-correct question, hide the middle fifth of the model's own reasoning, then
swap the hidden tokens for random ones. If hiding works, the answer score is unchanged by the
swap. The same swap without hiding must change the score, or the check proves nothing.
Also prints the attention-mask and rotary lines of the model's source for a manual audit.
"""

import argparse
import inspect
import random
import re

import torch

from dlm.config import MODELS
from dlm.generate import load_both_correct, load_traces
from dlm.models import Scorer

TOLERANCE = 1e-3


def audit_source(scorer):
    path = inspect.getsourcefile(type(scorer.model))
    print(f"source: {path}")
    pattern = re.compile(r"attn_mask|attention_mask|attention_bias|position_ids|rotary|is_causal")
    for i, line in enumerate(open(path).read().splitlines(), 1):
        if pattern.search(line):
            print(f"  {i:5d}: {line.strip()}")


def random_tokens(scorer, n, rng):
    special = set(scorer.tok.all_special_ids)
    out = []
    while len(out) < n:
        t = rng.randrange(len(scorer.tok))
        if t not in special:
            out.append(t)
    return out


def check_model(name, qids, n, audit):
    print(f"\n===== {name} =====")
    scorer = Scorer(name)
    if audit:
        audit_source(scorer)
    traces = load_traces(name, "train")
    rng = random.Random(0)
    worst_hidden, swap_effects, hide_effects = 0.0, [], []
    for qid in qids[:n]:
        r = traces[qid]
        item = scorer.build(r["question"], r["reasoning"], r["gold"])
        k = len(item.reasoning)
        start, end = 2 * k // 5, 3 * k // 5
        swapped = scorer.build(r["question"], r["reasoning"], r["gold"])
        swapped.reasoning[start:end] = random_tokens(scorer, end - start, rng)

        base = scorer.score(item)["logprob"]
        hidden = scorer.score(item.hide_reasoning(start, end))["logprob"]
        hidden_swapped = scorer.score(swapped.hide_reasoning(start, end))["logprob"]
        visible_swapped = scorer.score(swapped)["logprob"]

        diff = abs(hidden - hidden_swapped)
        worst_hidden = max(worst_hidden, diff)
        swap_effects.append(abs(base - visible_swapped))
        hide_effects.append(abs(base - hidden))
        print(
            f"  q{qid} span={start}:{end}/{k} base={base:.4f} hidden={hidden:.4f} "
            f"hidden+swap={hidden_swapped:.4f} swap_visible={visible_swapped:.4f} diff={diff:.2e}"
        )

    print(f"max |hidden - hidden+swap| = {worst_hidden:.2e} (must be < {TOLERANCE})")
    print(f"mean |base - swap_visible| = {sum(swap_effects) / len(swap_effects):.4f} (must be > 0)")
    print(f"mean |base - hidden|       = {sum(hide_effects) / len(hide_effects):.4f}")
    ok = worst_hidden < TOLERANCE and max(swap_effects) > TOLERANCE
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    scorer.unload()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)
    qids = load_both_correct("train")
    results = {m: check_model(m, qids, args.n, not args.no_audit) for m in args.models}
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
