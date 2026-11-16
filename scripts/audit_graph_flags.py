"""Summarize structural/semantic graph flags from results/steps (no API, no GPU).

Writes reports/graph_audit_flags.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dlm.config import PROJ, STEPS_DIR


def main():
    lines = [
        "# Graph flag audit",
        "",
        "Auto summary after thinned verifier (structure = hard fail; semantics = flags).",
        "",
    ]
    for model in ("llada", "qwen"):
        path = STEPS_DIR / f"{model}_test.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open()]
        ok = sum(r["ok"] for r in rows)
        final = sum(r.get("final_step") is not None for r in rows)
        flag_counts = Counter()
        for r in rows:
            for f in r.get("flags") or []:
                key = f.split(":")[0]
                if "possible_missed_computation" in f:
                    key = "possible_missed_computation"
                elif f.startswith("arith") or "arith " in f:
                    key = "arith_mismatch"
                flag_counts[key] += 1
        cov = [r["coverage_frac"] for r in rows if r.get("coverage_frac") is not None]
        cov.sort()
        med = cov[len(cov) // 2] if cov else None
        lines += [
            f"## {model}",
            "",
            f"- n={len(rows)}, ok={ok}, with final_step={final}",
            f"- median coverage_frac={med:.3f}" if med is not None else "- no coverage",
            "",
            "Flag counts (instances, not unique traces):",
            "",
        ]
        for k, v in flag_counts.most_common():
            lines.append(f"- `{k}`: {v}")
        n_miss = sum(
            1
            for r in rows
            if any("possible_missed_computation" in f for f in (r.get("flags") or []))
        )
        n_low = sum(
            1 for r in rows if any(str(f).startswith("low_coverage") for f in (r.get("flags") or []))
        )
        n_no = sum(
            1 for r in rows if "answer not in reasoning" in (r.get("flags") or [])
        )
        lines += [
            "",
            f"Unique traces: missed_comp={n_miss}, low_coverage={n_low}, answer_not_in_reasoning={n_no}",
            "",
        ]

    tax = PROJ / "reports" / "missing_gold_taxonomy.json"
    if tax.exists():
        data = json.loads(tax.read_text())
        lines += ["## Missing-gold taxonomy (from strip rebuild)", ""]
        for model, items in data.items():
            by = Counter(x["label"] for x in items)
            lines.append(f"- **{model}**: " + ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
        lines.append("")

    out = PROJ / "reports" / "graph_audit_flags.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
