#!/usr/bin/env python
"""Compare the LLM judge's content_type labels against the rule-based labels.

    python scripts/compare_content_type.py judge_output.jsonl

Joins the judge output (segment_id, content_type[, confidence, rationale]) to the
rule labels in data/derived/segments.jsonl, then reports overall agreement,
Cohen's kappa, a confusion table (rule -> judge), and every disagreement for human
review. The rule labels are NOT ground truth; disagreements are candidates to
inspect, not automatic corrections.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["clinical_exposition", "somatic_symptoms", "risk_suicide", "criteria",
          "case_example", "dosage", "referral_criteria"]


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _kappa(rule: list[str], judge: list[str]) -> float:
    n = len(rule)
    po = sum(a == b for a, b in zip(rule, judge)) / n
    cr, cj = Counter(rule), Counter(judge)
    pe = sum((cr[k] / n) * (cj[k] / n) for k in set(cr) | set(cj))
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: compare_content_type.py <judge_output.jsonl>")
    judge_rows = {r["segment_id"]: r for r in _load(Path(sys.argv[1]))}
    rule_rows = {r["segment_id"]: r for r in _load(ROOT / "data" / "derived" / "segments.jsonl")}

    common = [sid for sid in rule_rows if sid in judge_rows]
    missing = [sid for sid in rule_rows if sid not in judge_rows]
    unknown = [sid for sid in judge_rows if sid not in rule_rows]

    rule = [rule_rows[s]["content_type"] for s in common]
    judge = [judge_rows[s]["content_type"] for s in common]
    bad = [judge_rows[s]["content_type"] for s in common if judge_rows[s]["content_type"] not in LABELS]

    agree = sum(a == b for a, b in zip(rule, judge))
    print(f"segments compared: {len(common)}  (missing from judge: {len(missing)}, "
          f"unknown ids in judge: {len(unknown)})")
    if bad:
        print(f"WARNING: judge used {len(bad)} out-of-vocabulary label(s): {sorted(set(bad))}")
    print(f"agreement: {agree}/{len(common)} = {agree/len(common):.1%}")
    print(f"Cohen's kappa: {_kappa(rule, judge):.3f}\n")

    conf: dict = defaultdict(Counter)
    for a, b in zip(rule, judge):
        conf[a][b] += 1
    print("confusion (rule -> judge), rows=rule label:")
    for a in LABELS:
        if conf[a]:
            row = ", ".join(f"{b}:{c}" for b, c in conf[a].most_common())
            print(f"  {a:20} -> {row}")

    print("\ndisagreements (inspect these):")
    for s in common:
        r, j = rule_rows[s]["content_type"], judge_rows[s]["content_type"]
        if r != j:
            conf_s = judge_rows[s].get("confidence", "?")
            why = judge_rows[s].get("rationale", "")
            print(f"  {s}  rule={r} | judge={j} ({conf_s})  {rule_rows[s]['heading_path'][:50]}")
            if why:
                print(f"      -> {why}")
    if missing:
        print(f"\nnot labeled by judge: {missing}")


if __name__ == "__main__":
    main()
