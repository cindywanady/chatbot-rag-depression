#!/usr/bin/env python
"""Build the psychologist reading packets from the frozen assignments.

One packet per P1 case: the patient question, the chatbot briefing, and the exact
k=5 passages the generator received. Correctness and completeness are judged by the
psychologist against the MoH guideline (no pre-made answer key). The internal risk
flag is deliberately not printed - BAGIAN 0 is the rater's own judgement - and
light-form packets carry a printed BAGIAN-3 skip notice so the rater never chooses
what to skip (design v2 sec.3).

Scoring happens in the psychologist workbooks, not here:
scripts/build_psychologist_workbooks.py owns the rating sheets, the blind response
codes and the sealed condition key.

    ./.venv/bin/python scripts/build_eval_packets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from depression_rag.evaluation.packets import (  # noqa: E402
    build_packet_content,
    render_markdown,
)
from depression_rag.evaluation.study_assignments import RATERS  # noqa: E402

EVAL = ROOT / "outputs" / "evaluation"


TEMPLATE_RULES = (
    "Aturan jawaban (kedua kondisi): gunakan template jawaban yang sama; tulis "
    "prosa Anda sendiri kepada penanya; JANGAN menyalin judul bagian, rujukan "
    "[n], atau kode pedoman (mis. MI.7) dari alat bantu.")



def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_default(assignments: dict, answers: dict[str, dict]) -> None:
    # Beside the workbooks that reference them. build_psychologist_workbooks.py
    # tells raters to "buka paket di folder p1_packets/", and the workbooks live
    # in study_kit/02_PSIKOLOG/ — so packets written to outputs/evaluation/
    # were one level outside the kit and, fatally, outside study_kit.zip. The
    # psychologists would have received rating sheets pointing at 36 files they
    # did not have. Fixed 2026-07-27.
    p1_dir = EVAL / "study_kit" / "02_PSIKOLOG" / "p1_packets"
    p1_dir.mkdir(parents=True, exist_ok=True)
    cases = assignments["cases"]

    for c in cases:
        if not c["in_p1"]:
            continue
        # Content assembled in the library so this file and the workbook sheet
        # (build_psychologist_workbooks.py) cannot drift apart — two raters
        # reading different material would void the P1 comparison.
        packet = build_packet_content(c, answers[c["question_id"]])
        (p1_dir / packet.file_name).write_text(render_markdown(packet), encoding="utf-8")

    n_p1 = sum(c["in_p1"] for c in cases)
    n_full = sum(c.get("p1_form") == "full" for c in cases)
    print(f"wrote {n_p1} P1 packets ({n_full} full / {n_p1 - n_full} light) -> {p1_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignments", default=str(ROOT / "data" / "derived" / "eval_case_assignments.json"))
    ap.add_argument("--answers", default=str(ROOT / "outputs" / "analysis" / "study_answers_chatbot.jsonl"))
    args = ap.parse_args(argv)

    assignments = json.loads(Path(args.assignments).read_text(encoding="utf-8"))
    answers = {r["question_id"]: r for r in _load(Path(args.answers))}
    build_default(assignments, answers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
