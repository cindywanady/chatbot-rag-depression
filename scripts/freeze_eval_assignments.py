#!/usr/bin/env python
"""Draw and FREEZE the human-evaluation case assignments (design v2).

Implements `documents/evaluation/evaluation_design_v2.md` §2, §2b and §3:

  * counselor sample (P2/K1): 24 cases = 6 risk + 6 non-risk per block,
    drawn with the project seed from the frozen 50-question study set;
  * crossover: counselor_1 answers the block-1 sample WITH the chatbot and
    the block-2 sample WITHOUT it; counselor_2 is the mirror image;
  * P1 set (psychologists rate the chatbot briefing directly): the union of
    the counselor sample and ALL risk-flagged cases -> 24 + 24 - 12 = 36;
  * §2b rater split: whole case bundles dealt to two psychologists with a
    16-item shared overlap (4 counselor + 4 risk-only cases) -> 50 items
    each (34 unique + 16 shared);
  * §3 faithfulness half: BAGIAN 3 on a stratified half of the packets
    (18 of 36); every shared packet is full-form.

Risk flags are the deployed safety screen's verdicts recorded at generation
time (`outputs/analysis/study_answers_chatbot.jsonl -> risky`), so the
assignment uses exactly the flags the study instrument uses.

The output is a FROZEN protocol artifact: the script refuses to overwrite an
existing assignment file unless --force is given, and records the seed, input
hashes and draw date so the sampling is provably pre-specified. All draws
consume one seeded generator in a fixed order (see
`depression_rag.evaluation.study_assignments`), so the counselor sample is
byte-identical to the original 2026-07-16 draw and the newer splits extend
it without re-drawing anything.

    ./.venv/bin/python scripts/freeze_eval_assignments.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from depression_rag.evaluation.study_assignments import (  # noqa: E402
    RATERS,
    assign_p1_forms,
    assign_raters,
    draw_counselor_sample,
    rater_summary,
)

SEED = 20260628                 # the single project seed
PER_BLOCK = {"risk": 6, "nonrisk": 6}   # counselor sample per block


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default=str(ROOT / "data" / "derived" / "questions_alodokter_sample50.jsonl"))
    ap.add_argument("--answers", default=str(ROOT / "outputs" / "analysis" / "study_answers_chatbot.jsonl"),
                    help="chatbot generation records; source of the risk flags")
    ap.add_argument("--out", default=str(ROOT / "data" / "derived" / "eval_case_assignments.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite an existing (frozen) assignment file")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"REFUSING to overwrite frozen assignments: {out}\n"
              "The sample must stay fixed once drawn (protocol). Use --force only "
              "if the study has NOT started and you intend to re-draw.")
        return 1

    q_path, a_path = Path(args.questions), Path(args.answers)
    questions = {r["question_id"]: r for r in map(json.loads, open(q_path, encoding="utf-8"))}
    risky = {r["question_id"]: bool(r["risky"]) for r in map(json.loads, open(a_path, encoding="utf-8"))}
    missing = sorted(set(questions) - set(risky))
    if missing:
        raise SystemExit(f"no risk flag for {len(missing)} questions (regenerate answers first): {missing[:5]}")

    cases = sorted(questions.values(), key=lambda r: r["study_id"])  # deterministic order
    rng = random.Random(args.seed)

    counselor_sample = set(draw_counselor_sample(cases, risky, rng, PER_BLOCK))
    all_risk = [r["question_id"] for r in cases if risky[r["question_id"]]]
    p1_ids = counselor_sample | set(all_risk)

    records = []
    for r in cases:
        qid = r["question_id"]
        in_counselor = qid in counselor_sample
        records.append({
            "study_id": r["study_id"],
            "question_id": qid,
            "block": r["block"],
            "risky": risky[qid],
            "in_counselor_sample": in_counselor,
            # crossover: counselor_1 = with chatbot on block 1, without on
            # block 2; counselor_2 mirrored
            "counselor_condition": (
                {"counselor_1": "chatbot" if r["block"] == 1 else "no_chatbot",
                 "counselor_2": "no_chatbot" if r["block"] == 1 else "chatbot"}
                if in_counselor else None),
            "in_p1": qid in p1_ids,
        })

    assign_raters(records, rng)      # §2b — consumes rng after the sample draw
    assign_p1_forms(records, rng)    # §3  — consumes rng last

    n_c = sum(r["in_counselor_sample"] for r in records)
    n_overlap = sum(r["in_counselor_sample"] and r["risky"] for r in records)
    n_p1 = sum(r["in_p1"] for r in records)
    summary = rater_summary(records)
    payload = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "design": "documents/evaluation/evaluation_design_v2.md §2, §2b, §3",
        "seed": args.seed,
        "inputs": {str(q_path.relative_to(ROOT)): _sha256(q_path),
                   str(a_path.relative_to(ROOT)): _sha256(a_path)},
        "counts": {"corpus": len(records), "counselor_sample": n_c,
                   "counselor_sample_risk": n_overlap, "all_risk": len(all_risk),
                   "p1_union": n_p1, "p1_full_form": summary["p1_full_total"],
                   "shared_items": summary["shared_items"],
                   "raters": {r: summary[r] for r in RATERS}},
        "cases": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"corpus {len(records)} | counselor sample {n_c} "
          f"({n_overlap} risk + {n_c - n_overlap} non-risk) | "
          f"risk cases {len(all_risk)} | P1 union {n_p1} "
          f"({summary['p1_full_total']} full / {n_p1 - summary['p1_full_total']} light)")
    for block in (1, 2):
        ids = [r["study_id"] for r in records if r["in_counselor_sample"] and r["block"] == block]
        print(f"  block {block} counselor cases: {' '.join(ids)}")
    shared = [r["study_id"] for r in records if r["p1_rater"] == "both"]
    print(f"shared cases (both raters): {' '.join(shared)} "
          f"-> {summary['shared_items']} shared items")
    for rater in RATERS:
        mine = [r["study_id"] for r in records if r["p1_rater"] == rater]
        s = summary[rater]
        print(f"{rater}: {s['items']} items ({s['p2_answers']} answers + "
              f"{s['p1_packets']} packets, {s['p1_full']} full-form) | "
              f"unique cases: {' '.join(mine)}")
    print(f"\nFROZEN -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
