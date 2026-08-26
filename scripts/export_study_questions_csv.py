#!/usr/bin/env python
"""One CSV holding every finalized study question and the briefing it produced.

The study's question text lives in two JSONL files and its case assignments in a
third, which makes "what exactly was asked, and what came back?" a three-file
join. This exports that join once, as a single spreadsheet-openable file, so the
finalized set can be read and checked without any tooling.

The `question` column is the text **as sent to the generator** — the same string
`generate_study_answers.py` fed to Gemma and the same string a participant sees.
Since 2026-07-27 the frozen files are de-identified, so there is no longer a
distinction between "stored" and "rendered": it is one text. (Before that date
there were two, and this export would have needed both columns.)

`doctor_answer` is NOT exported by default. It is the Alodokter GP's reply,
carried as provenance only — nothing scores against it since
`doctor_point_coverage` was dropped on 2026-07-27, and it is a second source of
patient names. Pass --include-doctor-answer if you specifically need it.

Replaces `data/derived/questions_sample50_export.csv`, an undated orphan no
script wrote, which still contained the raw names.

    .venv/bin/python scripts/export_study_questions_csv.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from depression_rag.observability import ManifestBuilder  # noqa: E402

Q_PATH = ROOT / "data" / "derived" / "questions_alodokter_sample50.jsonl"
A_PATH = ROOT / "outputs" / "analysis" / "study_answers_chatbot.jsonl"
C_PATH = ROOT / "data" / "derived" / "eval_case_assignments.json"
OUT = ROOT / "outputs" / "analysis" / "study_questions_and_answers.csv"

COLUMNS = [
    "study_id", "question_id", "block",
    "risky", "risk_reason",
    "in_counselor_sample", "in_p1", "p1_form",
    "question_title", "question", "question_chars",
    "retrieved_chunk_ids", "retrieved_pages",
    "generated_answer", "answer_chars", "gen_seconds",
    "source_sha256",
]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--include-doctor-answer", action="store_true",
                    help="also export the GP forum reply (provenance only; carries names)")
    args = ap.parse_args(argv)

    questions = {r["question_id"]: r for r in _jsonl(Q_PATH)}
    answers = {r["study_id"]: r for r in _jsonl(A_PATH)}
    cases = {c["study_id"]: c for c in json.loads(C_PATH.read_text(encoding="utf-8"))["cases"]}

    missing = set(answers) ^ set(cases)
    if missing:
        raise SystemExit(f"answers and assignments disagree on cases: {sorted(missing)}")

    columns = list(COLUMNS)
    if args.include_doctor_answer:
        columns.insert(columns.index("generated_answer"), "doctor_answer")

    rows = []
    for sid in sorted(answers, key=lambda s: int(s[1:])):
        a, c = answers[sid], cases[sid]
        q = questions.get(a["question_id"], {})
        ctx = a.get("retrieved_context") or []
        row = {
            "study_id": sid,
            "question_id": a["question_id"],
            "block": c.get("block"),
            "risky": a.get("risky"),
            "risk_reason": a.get("risk_reason") or "",
            "in_counselor_sample": c.get("in_counselor_sample"),
            "in_p1": c.get("in_p1"),
            "p1_form": c.get("p1_form") or "",
            "question_title": q.get("question_title", ""),
            "question": a["question"],
            "question_chars": len(a["question"]),
            "retrieved_chunk_ids": " | ".join(str(x.get("chunk_id", "")) for x in ctx),
            "retrieved_pages": " | ".join(str(x.get("pages", "")) for x in ctx),
            "generated_answer": a["generated_answer"],
            "answer_chars": len(a["generated_answer"]),
            "gen_seconds": a.get("gen_seconds"),
            "source_sha256": q.get("source_sha256", ""),
        }
        if args.include_doctor_answer:
            row["doctor_answer"] = a.get("doctor_answer", "")
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel misreads plain UTF-8 and mangles the Indonesian text
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    run_id = f"export_study_questions_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    m = ManifestBuilder(run_id=run_id, description="study question/answer CSV export")
    for label, p in (("questions", Q_PATH), ("answers", A_PATH), ("assignments", C_PATH)):
        m.add_input(label, p)
    m.add_output(out)
    m.record("n_rows", len(rows))
    m.record("includes_doctor_answer", args.include_doctor_answer)
    mp = m.write(ROOT / "outputs" / "manifests" / f"{run_id}.json")

    risky = sum(1 for r in rows if r["risky"] is True)
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"  {len(rows)} rows · {len(columns)} columns")
    print(f"  {risky} risk-flagged · {sum(1 for r in rows if r['in_counselor_sample'])} in counselor sample "
          f"· {sum(1 for r in rows if r['in_p1'])} in P1")
    print(f"  manifest {mp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
