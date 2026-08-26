"""Validate and summarise a generated question set.

Used for both the LLM-drafted candidates and the human-validated gold set. It
enforces the prompt's output schema, checks that every referenced passage exists,
and that difficulty/passage-count are consistent (multi_hop needs >=2 passages;
factoid/applied_case need exactly 1). It also reports the taxonomy / difficulty /
gold-concept distributions for the write-up.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_TYPES = {
    "diagnostic_criteria",
    "symptom_recognition",
    "pharmacotherapy_dosing",
    "referral_criteria",
    "risk_suicide_emergency",
    "psychoeducation",
    "special_populations",
    "differential_comorbidity",  # added beyond the base 7-type taxonomy (own gold concept)
}
DIFFICULTIES = {"factoid", "multi_hop", "applied_case"}
REQUIRED_FIELDS = (
    "question_id", "question", "question_type",
    "difficulty", "passage_ids", "reference_answer",
)


@dataclass
class QuestionValidation:
    n: int = 0
    errors: list[str] = field(default_factory=list)
    by_type: dict[str, int] = field(default_factory=dict)
    by_difficulty: dict[str, int] = field(default_factory=dict)
    by_concept: dict[str, int] = field(default_factory=dict)
    passages_covered: int = 0
    passages_total: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def load_questions(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def validate_questions(questions: list[dict],
                       concept_by_passage: dict[str, str]) -> QuestionValidation:
    """Validate questions against the set of known gold passage ids.

    `concept_by_passage` maps gold passage_id -> concept (from gold_passages).
    """
    v = QuestionValidation(n=len(questions), passages_total=len(concept_by_passage))
    seen_ids: set[str] = set()
    covered: set[str] = set()

    for i, q in enumerate(questions):
        tag = q.get("question_id", f"<index {i}>")

        missing = [f for f in REQUIRED_FIELDS if f not in q]
        if missing:
            v.errors.append(f"{tag}: missing fields {missing}")
            continue

        qid = q["question_id"]
        if qid in seen_ids:
            v.errors.append(f"{tag}: duplicate question_id")
        seen_ids.add(qid)

        if q["question_type"] not in QUESTION_TYPES:
            v.errors.append(f"{tag}: bad question_type {q['question_type']!r}")
        if q["difficulty"] not in DIFFICULTIES:
            v.errors.append(f"{tag}: bad difficulty {q['difficulty']!r}")

        pids = q["passage_ids"]
        if not isinstance(pids, list) or not pids:
            v.errors.append(f"{tag}: passage_ids must be a non-empty list")
            pids = []
        unknown = [p for p in pids if p not in concept_by_passage]
        if unknown:
            v.errors.append(f"{tag}: unknown passage_ids {unknown}")

        diff = q["difficulty"]
        if diff == "multi_hop" and len(pids) < 2:
            v.errors.append(f"{tag}: multi_hop needs >=2 passage_ids, got {len(pids)}")
        if diff in ("factoid", "applied_case") and len(pids) != 1:
            v.errors.append(f"{tag}: {diff} needs exactly 1 passage_id, got {len(pids)}")

        if not str(q["question"]).strip():
            v.errors.append(f"{tag}: empty question")
        if not str(q["reference_answer"]).strip():
            v.errors.append(f"{tag}: empty reference_answer")

        covered.update(p for p in pids if p in concept_by_passage)

    v.by_type = dict(Counter(q.get("question_type") for q in questions))
    v.by_difficulty = dict(Counter(q.get("difficulty") for q in questions))
    concept_counter: Counter = Counter()
    for q in questions:
        for p in q.get("passage_ids", []):
            if p in concept_by_passage:
                concept_counter[concept_by_passage[p]] += 1
    v.by_concept = dict(concept_counter)
    v.passages_covered = len(covered)
    return v
