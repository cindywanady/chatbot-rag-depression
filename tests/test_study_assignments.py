"""Study-assignment draws: counts, stratification, bundle and form invariants.

Runs on a synthetic 50-case corpus shaped like the frozen study set
(2 blocks x 25, 12 risk per block), so no data files are needed.
"""

import random

from depression_rag.evaluation.study_assignments import (
    RATERS,
    assign_p1_forms,
    assign_raters,
    draw_counselor_sample,
    rater_summary,
    BLOCKS,
    PER_STRATUM_FULL,
)

PER_BLOCK = {"risk": 6, "nonrisk": 6}
SEED = 20260628


def _corpus():
    cases, risky = [], {}
    for i in range(1, 51):
        block = 1 if i <= 25 else 2
        qid = f"q{i:03d}"
        cases.append({"study_id": f"Q{i:02d}", "question_id": qid, "block": block})
        risky[qid] = (i - 1) % 25 < 12          # 12 risk cases per block
    return cases, risky


def _assign(seed=SEED):
    cases, risky = _corpus()
    rng = random.Random(seed)
    sample = set(draw_counselor_sample(cases, risky, rng, PER_BLOCK))
    all_risk = {q for q, r in risky.items() if r}
    records = [{**c, "risky": risky[c["question_id"]],
                "in_counselor_sample": c["question_id"] in sample,
                "in_p1": c["question_id"] in (sample | all_risk)}
               for c in cases]
    assign_raters(records, rng)
    assign_p1_forms(records, rng)
    return records


def test_counselor_sample_stratification():
    records = _assign()
    sample = [r for r in records if r["in_counselor_sample"]]
    assert len(sample) == 24
    for block in (1, 2):
        for is_risk, n in ((True, 6), (False, 6)):
            assert sum(r["block"] == block and r["risky"] is is_risk
                       for r in sample) == n


def test_p1_union_is_36():
    records = _assign()
    assert sum(r["in_p1"] for r in records) == 36
    # every risk case and every counselor case is P1-rated
    assert all(r["in_p1"] for r in records if r["risky"] or r["in_counselor_sample"])


def test_rater_split_counts_and_balance():
    records = _assign()
    summary = rater_summary(records)
    assert summary["shared_items"] == 16
    shared = [r for r in records if r["p1_rater"] == "both"]
    assert len(shared) == 8
    assert sum(r["in_counselor_sample"] for r in shared) == 4
    for rater in RATERS:
        s = summary[rater]
        assert (s["counselor_cases"], s["risk_only_cases"]) == (14, 8)
        assert (s["p2_answers"], s["p1_packets"], s["items"]) == (28, 22, 50)
        mine_counselor = [r for r in records if r["p1_rater"] in ("both", rater)
                          and r["in_counselor_sample"]]
        assert sum(r["risky"] for r in mine_counselor) == 7
        assert sum(r["block"] == 1 for r in mine_counselor) == 7
    # unique sets are disjoint and cover all of P1
    unique = [r["p1_rater"] for r in records if r["in_p1"]]
    assert unique.count("both") + unique.count(RATERS[0]) + unique.count(RATERS[1]) == 36
    # bundle rule: every P1 case has a rater, nothing else does
    assert all((r["p1_rater"] is not None) == r["in_p1"] for r in records)


def test_faithfulness_full_form_stratified_and_covers_shared():
    """The full-form set = every shared packet + PER_STRATUM_FULL per block x stratum.

    Asserted as that rule rather than as fixed totals, because the count is a
    tuning parameter: it was cut from 2/2/1 per block to 1/1/1 on 2026-07-30 to
    save rater time, and a test pinned to "18 full" would have to be rewritten
    every time instead of checking that the design still holds.
    """
    records = _assign()
    p1 = [r for r in records if r["in_p1"]]
    full = [r for r in p1 if r["p1_form"] == "full"]
    shared = [r for r in p1 if r["p1_rater"] == "both"]

    # every packet has a form, and the two sets partition P1
    assert all(r["p1_form"] in ("full", "light") for r in p1)
    assert len(full) + sum(r["p1_form"] == "light" for r in p1) == len(p1)

    # THE load-bearing one: inter-rater agreement must cover the faithfulness
    # items, which is only possible if every shared packet is full-form
    assert all(r["p1_form"] == "full" for r in shared)

    # the drawn half: exactly PER_STRATUM_FULL per (block x stratum)
    def stratum(r):
        if r["in_counselor_sample"]:
            return "counselor_risk" if r["risky"] else "counselor_nonrisk"
        return "risk_only"

    drawn = [r for r in full if r["p1_rater"] != "both"]
    assert len(drawn) == len(BLOCKS) * 3 * PER_STRATUM_FULL
    for block in BLOCKS:
        for name in ("counselor_risk", "counselor_nonrisk", "risk_only"):
            n = sum(r["block"] == block and stratum(r) == name for r in drawn)
            assert n == PER_STRATUM_FULL, f"block {block}/{name}: {n}"
    assert len(full) == len(shared) + len(drawn)


def test_deterministic_and_seed_sensitive():
    a, b = _assign(), _assign()
    assert a == b
    c = _assign(seed=1)
    assert a != c


def test_counselor_draw_unchanged_by_later_draws():
    """Appending the rater/form draws must not disturb the frozen sample."""
    cases, risky = _corpus()
    alone = draw_counselor_sample(cases, risky, random.Random(SEED), PER_BLOCK)
    full_run = [r["question_id"] for r in _assign() if r["in_counselor_sample"]]
    assert sorted(alone) == sorted(full_run)
