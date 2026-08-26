#!/usr/bin/env python
"""Audit the LLM risk classifier WITHOUT regenerating any study answer.

Re-runs only `llm_risk_check()` over the study questions, N times each, and
records the raw verdict strings. The current frozen file cannot answer these
questions itself: it was regenerated with `--pin-risk-from`, so the LLM stage
never ran and `risk_llm_verdict` is null for all 50 rows. (The pre-pin verdict
text survives in `archive/pre_prompt_fix_20260801-115847/`.)

  1. *Negation false positives* — the production parse is `"RISIKO" in raw.upper()`
     and "RISIKO" is a substring of "BERISIKO", so a reply like "tidak berisiko"
     is scored as a POSITIVE flag. Repeats surface whether the model ever phrases
     it that way.
  2. *Agreement* — how often the re-run disagrees with the frozen flag in
     `outputs/analysis/study_answers_chatbot.jsonl`. This is the substantive
     check, and it is what should be reported.

On repeats: `llm_risk_check` decodes greedily (temperature 0.0, overriding
`cfg.temperature`), so identical repeats are close to expected and a zero flip
count evidences a stable serving stack rather than a robust classifier. Repeats
are still worth running — they widen the sample of verdict *phrasings* fed to the
negation check above — but do not report the flip count as a robustness result.

Nothing downstream is touched: no briefing is regenerated, no packet rebuilt,
no assignment re-frozen. Output goes to its own file.

    .venv-chat/bin/python scripts/audit_risk_verdicts.py --repeats 5

Mirrors production control flow exactly (`ChatEngine._prepare`): the keyword
screen runs first and the LLM is consulted ONLY when no keyword matched, so a
keyword-flagged case is recorded with `verdicts: []`, as in the real system.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from depression_rag.chatbot import (  # noqa: E402
    LLM_RISK_PROMPT,
    build_generator,
    load_chatbot_config,
    risk_screen,
)

# A verdict is a negation-style false positive when the production substring
# parse says "risk" but the text is plainly a denial.
NEGATIONS = ("tidak", "bukan", "no risk", "not at risk")


def is_negation_fp(raw: str) -> bool:
    """Production parse says "risk", but the text plainly denies it.

    Covers both shapes seen in practice: a denial ("tidak berisiko" — RISIKO is a
    substring of BERISIKO) and a hedge that also names the safe label
    ("RISIKO: AMAN").
    """
    up, low = raw.upper(), raw.lower()
    if "RISIKO" not in up:
        return False
    return any(t in low for t in NEGATIONS) or "AMAN" in up


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/chatbot.yaml"))
    ap.add_argument("--questions", default=str(ROOT / "data/derived/questions_alodokter_sample50.jsonl"))
    ap.add_argument("--frozen", default=str(ROOT / "outputs/analysis/study_answers_chatbot.jsonl"),
                    help="the frozen run, for flag comparison (read-only)")
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis/risk_verdicts_audit.jsonl"))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = load_chatbot_config(args.config)
    generator = build_generator(cfg)   # generator only — no index, no retrieval

    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8")]
    frozen = {r["question_id"]: r for r in map(json.loads, open(args.frozen, encoding="utf-8"))}

    # Report the temperature the risk call actually uses, not cfg.temperature:
    # llm_risk_check hard-codes 0.0 and ignores the config value.
    print(f"[audit] {len(questions)} questions x {args.repeats} repeats, "
          f"model={cfg.checkpoint}, temperature=0.0 (greedy; "
          f"cfg.temperature={cfg.temperature} does not apply to the risk stage)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()

    for i, q in enumerate(questions, 1):
        matched = risk_screen(q["question"], cfg.safety_keywords)
        verdicts: list[str] = []
        flags: list[bool] = []
        if not matched and cfg.llm_check:      # same gate as ChatEngine._prepare
            for _ in range(args.repeats):
                flag, raw = generator.llm_risk_check(q["question"])
                verdicts.append(raw)
                flags.append(flag)
        fz = frozen.get(q["question_id"], {})
        row = {
            "study_id": q.get("study_id"),
            "question_id": q["question_id"],
            "keyword_matches": matched,
            "llm_verdicts": verdicts,
            "llm_flags": flags,
            "n_risk_of_repeats": sum(flags),
            "unstable": bool(flags) and 0 < sum(flags) < len(flags),
            "negation_false_positive": [v for v in verdicts if is_negation_fp(v)],
            # majority of the repeats, combined with the keyword screen as in production
            "rerun_risky": bool(matched) or (sum(flags) * 2 > len(flags) if flags else False),
            "frozen_risky": fz.get("risky"),
            "frozen_reason": fz.get("risk_reason"),
        }
        row["agrees_with_frozen"] = row["rerun_risky"] == row["frozen_risky"]
        rows.append(row)
        mark = "kw" if matched else f"{sum(flags)}/{len(flags)}"
        print(f"  [{i:>2}/{len(questions)}] {row['study_id']} {mark:>5} "
              f"{'' if row['agrees_with_frozen'] else '  <-- DISAGREES with frozen'}")

    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    llm_rows = [r for r in rows if r["llm_verdicts"]]
    unstable = [r for r in llm_rows if r["unstable"]]
    negfp = [r for r in rows if r["negation_false_positive"]]
    disagree = [r for r in rows if not r["agrees_with_frozen"]]
    shapes = collections.Counter(v.strip() for r in llm_rows for v in r["llm_verdicts"])

    print(f"\n=== summary  ({time.perf_counter() - t0:.0f}s) ===")
    print(f"questions            : {len(rows)}  ({len(rows) - len(llm_rows)} keyword-flagged, "
          f"{len(llm_rows)} sent to the classifier)")
    # Greedy decoding makes a zero here close to expected — it evidences a stable
    # serving stack, not a robust classifier. Do not report it as robustness.
    print(f"flips across {args.repeats}x     : {len(unstable)}  "
          f"{[r['study_id'] for r in unstable]}  (greedy decode; not a robustness result)")
    print(f"negation false pos.  : {len(negfp)}  {[r['study_id'] for r in negfp]}")
    print(f"disagrees with frozen: {len(disagree)}  {[r['study_id'] for r in disagree]}")
    print(f"re-run risky total   : {sum(r['rerun_risky'] for r in rows)} "
          f"(frozen: {sum(bool(r['frozen_risky']) for r in rows)})")
    print("\nverdict strings returned:")
    for text, n in shapes.most_common(12):
        print(f"  {n:>4}  {text!r}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
