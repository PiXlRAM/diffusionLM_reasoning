"""Extract and check the computation graph of every trace with an LLM.

Each call's raw output is cached per trace (keyed by the extractor model, reasoning effort,
instructions, schema and trace text), so reruns never call the API twice. Failed checks are
sent back for a corrected annotation, up to STEP_EXTRACTOR["max_attempts"] calls per trace.

Writes results/steps/{model}_{split}.jsonl and a readable review file
results/steps/review_{split}.md. Needs OPENAI_API_KEY.

Trial (read the review file before scaling up):
  python -m scripts.extract_steps --split train --n 10
"""

import argparse
import json
import time

from dlm.cache import make_key
from dlm.config import STEP_EXTRACTOR, STEPS_DIR
from dlm.generate import load_both_correct, load_traces
from dlm.steps import INSTRUCTIONS, SCHEMA, user_message, verify


def call(client, model, effort, messages):
    kwargs = {
        "model": model,
        "messages": messages,
        "reasoning_effort": effort,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "steps", "strict": True, "schema": SCHEMA},
        },
    }
    if effort == "none":
        kwargs["temperature"] = 0
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(f"model refused: {msg.refusal}")
    if not msg.content:
        raise RuntimeError("empty model response")
    return msg.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def extract(client, model, effort, trace, raw_dir, usage):
    key = make_key({
        "model": model,
        "effort": effort,
        "instructions": INSTRUCTIONS,
        "schema": SCHEMA,
        "question": trace["question"],
        "reasoning": trace["reasoning"],
    })
    path = raw_dir / f"{key}.json"
    if path.exists():
        attempts = json.loads(path.read_text())
    else:
        messages = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user_message(trace["question"], trace["reasoning"])},
        ]
        attempts = []
        for _ in range(STEP_EXTRACTOR["max_attempts"]):
            content, n_in, n_out = call(client, model, effort, messages)
            steps = json.loads(content)["steps"]
            check = verify(trace["question"], trace["reasoning"], trace["gold"], steps)
            attempts.append({
                "steps": steps, "tokens_in": n_in, "tokens_out": n_out, "errors": check["errors"],
            })
            usage["in"] += n_in
            usage["out"] += n_out
            if check["ok"]:
                break
            messages += [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your annotation failed these checks:\n- "
                        + "\n- ".join(check["errors"])
                        + "\nReturn a corrected annotation of the whole solution."
                    ),
                },
            ]
        path.write_text(json.dumps(attempts))
    steps = attempts[-1]["steps"]
    check = verify(trace["question"], trace["reasoning"], trace["gold"], steps)
    return steps, check, len(attempts)


def review_entry(model, qid, trace, steps, check):
    lines = [
        f"## {model} q{qid}  (gold {trace['gold']})",
        "",
        f"**Question:** {trace['question']}",
        "",
        "**Reasoning:**",
        "",
        "```",
        trace["reasoning"],
        "```",
        "",
    ]
    for i, s in enumerate(steps):
        tags = []
        if i == check["final_step"]:
            tags.append("FINAL")
        if i in check["chain"]:
            tags.append("chain")
        kind = "explicit" if s.get("equation_explicit") else "result-only"
        line = (
            f"{i}. **{s['role']}** ({kind})"
            f"{' [' + ', '.join(tags) + ']' if tags else ''}: \"{s['quote']}\""
        )
        if s.get("expression"):
            round_tag = f" {s['rounding']}" if s.get("rounding") else ""
            line += f"  \n   `{s['expression']} = {s['result']}`{round_tag}"
        elif s["result"] is not None:
            label = "repeats" if s["role"] == "restatement" else "states"
            line += f"  \n   {label} {s['result']}"
        srcs = [
            f"{x['value']} from {'step ' + str(x['step']) if x['source'] == 'step' else x['source']}"
            for x in s["inputs"]
        ]
        if srcs:
            line += f"  \n   inputs: {', '.join(srcs)}"
        lines.append(line)
    if check["errors"]:
        lines += ["", "**Errors:** " + " | ".join(check["errors"])]
    if check["flags"]:
        lines += ["", "**Flags:** " + " | ".join(check["flags"])]
    lines += ["", "Review: steps correct? roles correct? links correct? (write notes here)", "", "---", ""]
    return "\n".join(lines)


def load_done(path, extractor, effort):
    """Resume rows written with the same extractor settings; archive a mismatched file."""
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open()]
    if any(r.get("extractor") != extractor or r.get("effort", effort) != effort for r in rows):
        path.replace(path.with_suffix(path.suffix + ".bak"))
        return {}
    return {r["qid"]: r for r in rows}


def reverify(split, models):
    """Re-run code checks on saved extractions. No API calls."""
    review = [f"# Step extraction review ({split}, re-verified)", ""]
    for model in models:
        traces = load_traces(model, split)
        path = STEPS_DIR / f"{model}_{split}.jsonl"
        rows = [json.loads(l) for l in path.open()]
        stats = {"ok": 0, "failed": 0, "became_ok": 0, "became_fail": 0, "no_answer": 0}
        out = []
        for rec in rows:
            before = rec["ok"]
            t = traces[rec["qid"]]
            check = verify(t["question"], t["reasoning"], t["gold"], rec["steps"])
            keep = {k: rec[k] for k in ("qid", "model", "extractor", "effort", "reasoning", "gold", "steps", "calls")
                    if k in rec}
            keep.update(check)
            out.append(keep)
            stats["ok" if check["ok"] else "failed"] += 1
            stats["became_ok"] += check["ok"] and not before
            stats["became_fail"] += (not check["ok"]) and before
            stats["no_answer"] += "answer not in reasoning" in check["flags"]
            review.append(review_entry(model, rec["qid"], t, rec["steps"], check))
        path.write_text("".join(json.dumps(r) + "\n" for r in out))
        print(f"{model}: {stats}")
    review_path = STEPS_DIR / f"review_{split}.md"
    review_path.write_text("\n".join(review), encoding="utf-8")
    print(f"review file: {review_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="+", default=["llada", "qwen"])
    parser.add_argument("--n", type=int, default=None, help="limit the number of questions")
    parser.add_argument("--extractor", default=STEP_EXTRACTOR["model"])
    parser.add_argument("--effort", default=STEP_EXTRACTOR["reasoning_effort"])
    parser.add_argument("--price-in", type=float, default=STEP_EXTRACTOR["price_in"],
                        help="USD per million input tokens")
    parser.add_argument("--price-out", type=float, default=STEP_EXTRACTOR["price_out"],
                        help="USD per million output tokens")
    parser.add_argument("--reverify", action="store_true",
                        help="re-check saved extractions with the current verifier; no API calls")
    args = parser.parse_args()

    if args.reverify:
        reverify(args.split, args.models)
        return

    from openai import OpenAI

    client = OpenAI()
    raw_dir = STEPS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    qids = load_both_correct(args.split)[: args.n]
    usage = {"in": 0, "out": 0}
    review = [
        f"# Step extraction review ({args.split}, extractor {args.extractor}, effort {args.effort})",
        "",
    ]
    start = time.time()
    n_new = 0
    for model in args.models:
        traces = load_traces(model, args.split)
        out_path = STEPS_DIR / f"{model}_{args.split}.jsonl"
        done = load_done(out_path, args.extractor, args.effort)
        stats = {"ok": 0, "failed": 0, "retried": 0, "no_answer": 0, "ambiguous": 0, "resumed": 0}
        with out_path.open("a") as out:
            for qid in qids:
                if qid in done:
                    rec = done[qid]
                    steps, check, n_calls = rec["steps"], rec, rec.get("calls", 1)
                    stats["resumed"] += 1
                else:
                    t = traces[qid]
                    steps, check, n_calls = extract(
                        client, args.extractor, args.effort, t, raw_dir, usage,
                    )
                    out.write(json.dumps({
                        "qid": qid, "model": model, "extractor": args.extractor, "effort": args.effort,
                        "reasoning": t["reasoning"], "gold": t["gold"],
                        "steps": steps, "calls": n_calls, **check,
                    }) + "\n")
                    out.flush()
                    n_new += 1
                stats["ok" if check["ok"] else "failed"] += 1
                stats["retried"] += n_calls > 1
                stats["no_answer"] += "answer not in reasoning" in check.get("flags", [])
                stats["ambiguous"] += any(l.get("ambiguous") for l in check.get("links", []))
                review.append(review_entry(model, qid, traces[qid], steps, check))
        print(f"{model}: {stats}")
    review_path = STEPS_DIR / f"review_{args.split}.md"
    review_path.write_text("\n".join(review), encoding="utf-8")

    dollars = (usage["in"] * args.price_in + usage["out"] * args.price_out) / 1e6
    per = dollars / n_new if n_new else 0.0
    print(f"new API tokens: {usage['in']} in, {usage['out']} out, ${dollars:.4f} "
          f"({n_new} new traces, ~${per:.4f}/trace; ~${per * 408:.2f} for 204x2 test); "
          f"{time.time() - start:.0f}s")
    print(f"review file: {review_path}")


if __name__ == "__main__":
    main()
