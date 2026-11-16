"""Rebuild reasoning from saved raw generations under cut_sentence vs keep_operation.

Does not call the models. Writes:
  results/traces/rebuilt/{model}_{split}_{mode}.jsonl
  reports/strip_sensitivity_preview.md  (diffs + missing-gold taxonomy)

Usage:
  python -m scripts.rebuild_reasoning --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlm.config import RESULTS_DIR, TRACES_DIR
from dlm.data import normalize_number, split_generation
from dlm.generate import load_both_correct, load_traces, trace_path
from dlm.steps import values_stated_in


def _same_num(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(normalize_number(str(a))) - float(normalize_number(str(b)))) < 1e-6
    except ValueError:
        return False


def gold_in_text(text, gold):
    return any(_same_num(v, gold) for v in values_stated_in(text or ""))


def looks_like_operation(text):
    if not text:
        return False
    t = text.lower()
    if any(op in text for op in ("+", "-", "*", "/", "×", "÷", "=")):
        return True
    return any(
        w in t
        for w in ("multiply", "divide", "subtract", "add ", "times", "percent", "hourly rate")
    )


def rebuild_one(raw, mode):
    reasoning, pred, removed = split_generation(raw, mode=mode)
    return {
        "reasoning": reasoning,
        "pred": pred,
        "answer_sentence": removed,
        "n_reasoning_tokens": None,  # filled by caller if needed
        "strip_mode": mode,
    }


def classify_missing(old_reason, new_reason, removed, gold):
    """Taxonomy for traces flagged answer-not-in-reasoning under the original strip."""
    in_old = gold_in_text(old_reason, gold)
    in_new = gold_in_text(new_reason, gold)
    in_removed = gold_in_text(removed, gold) or looks_like_operation(removed)
    if not in_old and in_new:
        return "strip_artifact"
    if not in_old and in_removed and not in_new:
        # Operation moved into reasoning but gold digits still only in box — still artifact of cut
        # if the removed sentence held the operation; else plan-like with box-only answer.
        return "strip_artifact" if looks_like_operation(removed) else "plan_only"
    if not in_old and not in_new:
        return "plan_only"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="+", default=["llada", "qwen"])
    args = parser.parse_args()

    out_root = TRACES_DIR / "rebuilt"
    out_root.mkdir(parents=True, exist_ok=True)
    both = set(load_both_correct(args.split))
    report = [
        "# Strip sensitivity preview",
        "",
        "Original mode: `cut_sentence` (drop whole boxed sentence).",
        "Revised mode: `keep_operation` (keep operation text; strip boxed answer / terminal glue).",
        "",
    ]
    taxonomy = {"llada": [], "qwen": []}

    for model in args.models:
        traces = load_traces(model, args.split)
        src = trace_path(model, args.split)
        steps_path = RESULTS_DIR / "steps" / f"{model}_{args.split}.jsonl"
        flagged = set()
        if steps_path.exists():
            for line in steps_path.open():
                rec = json.loads(line)
                if "answer not in reasoning" in rec.get("flags", []):
                    flagged.add(rec["qid"])

        rebuilt_rows = {mode: [] for mode in ("cut_sentence", "keep_operation")}
        changed = []
        for qid, row in sorted(traces.items()):
            raw = row.get("raw") or ""
            if not raw:
                continue
            for mode in rebuilt_rows:
                parts = rebuild_one(raw, mode)
                new_row = {
                    **{k: row[k] for k in row if k not in parts},
                    **parts,
                    "qid": qid,
                    "source_trace": src.name,
                }
                rebuilt_rows[mode].append(new_row)
            old_r = split_generation(raw, mode="cut_sentence")[0]
            new_r = split_generation(raw, mode="keep_operation")[0]
            if old_r != new_r:
                changed.append(qid)

            if qid in flagged and qid in both:
                removed = split_generation(raw, mode="cut_sentence")[2]
                label = classify_missing(old_r, new_r, removed, row["gold"])
                taxonomy[model].append(
                    {
                        "qid": qid,
                        "label": label,
                        "gold": row["gold"],
                        "old_reasoning": old_r,
                        "new_reasoning": new_r,
                        "removed": removed,
                    }
                )

        for mode, rows in rebuilt_rows.items():
            path = out_root / f"{model}_{args.split}_{mode}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            print(f"wrote {path} ({len(rows)} rows)")

        report += [
            f"## {model}",
            "",
            f"- Source: `{src.name}`",
            f"- Reasoning text changed under keep_operation: **{len(changed)}** / {len(traces)}",
            f"- Both-correct missing-gold flags: **{len(taxonomy[model])}**",
            "",
        ]
        by = {}
        for item in taxonomy[model]:
            by.setdefault(item["label"], []).append(item["qid"])
        for label, qids in sorted(by.items()):
            report.append(f"- `{label}`: {len(qids)} — qids {qids}")
        report.append("")
        for item in taxonomy[model]:
            report += [
                f"### {model} q{item['qid']} — `{item['label']}` (gold {item['gold']})",
                "",
                "**Removed under cut_sentence:**",
                "",
                "```",
                item["removed"] or "(empty)",
                "```",
                "",
                "**Old reasoning (tail):**",
                "",
                "```",
                (item["old_reasoning"] or "")[-500:],
                "```",
                "",
                "**New reasoning (tail):**",
                "",
                "```",
                (item["new_reasoning"] or "")[-500:],
                "```",
                "",
                "---",
                "",
            ]

    from dlm.config import PROJ

    report_path = PROJ / "reports" / "strip_sensitivity_preview.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {report_path}")

    # Compact JSON taxonomy for later analysis
    tax_path = PROJ / "reports" / "missing_gold_taxonomy.json"
    tax_path.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    print(f"wrote {tax_path}")


if __name__ == "__main__":
    main()
