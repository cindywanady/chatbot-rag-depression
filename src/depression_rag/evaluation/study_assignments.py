"""Seeded draws for the human-evaluation study assignments.

Implements `documents/evaluation/evaluation_design_v2.md`:

  * §2  — counselor sample (24 cases = 6 risk + 6 non-risk per block) and the
          P1 union (counselor sample ∪ all risk cases = 36);
  * §2b — the two-psychologist split by whole case bundles: a 16-item shared
          overlap (4 counselor cases + 4 risk-only cases, stratified) and the
          remaining cases dealt alternately within each stratum → 50 items
          per rater (34 unique + 16 shared);
  * §3  — the half-faithfulness form assignment: BAGIAN 3 is completed for a
          stratified half of the packets (18 of 36 = 6 counselor-risk +
          6 counselor-non-risk + 6 risk-only, balanced across blocks), with
          every shared packet on the full form so inter-rater agreement
          covers the faithfulness items.

All functions consume the caller's `random.Random` in a FIXED, documented
order, so a single seeded generator reproduces the whole assignment — and
appending new draws never disturbs earlier ones. `draw_counselor_sample`
consumes first (4 sample calls) and must stay byte-compatible with the
original 2026-07-16 freeze.
"""

from __future__ import annotations

import random

RATERS = ("psychologist_1", "psychologist_2")
BLOCKS = (1, 2)

# Alternating deal starts per stratum, chosen so each rater ends up with
# 10 unique counselor cases balanced 5 risk / 5 non-risk and 5 per block
# (stratum size 5 is odd, so a fixed single start would skew 12/8).
_COUNSELOR_DEAL_STARTS = {(1, True): 0, (1, False): 1, (2, True): 1, (2, False): 0}
_RISKONLY_DEAL_STARTS = {1: 0, 2: 1}


def _sorted_ids(records: list[dict], **match) -> list[str]:
    """question_ids of records matching all keyword filters, in study_id order."""
    out = [r for r in records if all(r[k] == v for k, v in match.items())]
    return [r["question_id"] for r in sorted(out, key=lambda r: r["study_id"])]


def draw_counselor_sample(cases: list[dict], risky: dict[str, bool],
                          rng: random.Random, per_block: dict[str, int]) -> list[str]:
    """§2 counselor sample. rng consumption: 4 sample calls in block/risk order.

    `cases` must be sorted by study_id before calling (the frozen order).
    """
    sample: list[str] = []
    for block in BLOCKS:
        for is_risk, n in ((True, per_block["risk"]), (False, per_block["nonrisk"])):
            pool = [r["question_id"] for r in cases
                    if r["block"] == block and risky[r["question_id"]] is is_risk]
            if len(pool) < n:
                raise SystemExit(
                    f"block {block}: need {n} {'risk' if is_risk else 'non-risk'} "
                    f"cases, only {len(pool)} available")
            sample.extend(rng.sample(pool, n))
    return sample


def _deal(ids: list[str], rng: random.Random, start: int) -> dict[str, str]:
    """Shuffle ids and deal them alternately to the two raters."""
    ids = list(ids)
    rng.shuffle(ids)
    return {qid: RATERS[(start + i) % 2] for i, qid in enumerate(ids)}


def assign_raters(records: list[dict], rng: random.Random) -> None:
    """§2b: set `p1_rater` on every record ("both" / a rater name / None).

    rng consumption order: shared counselor-risk per block (2 × sample-1),
    shared counselor-non-risk per block (2 × sample-1), shared risk-only per
    block (2 × sample-2), then one shuffle per counselor stratum (4) and one
    per risk-only block (2).
    """
    for r in records:
        r["p1_rater"] = None

    by_id = {r["question_id"]: r for r in records}
    shared: list[str] = []
    for is_risk in (True, False):                     # shared counselor cases
        for block in BLOCKS:
            pool = _sorted_ids(records, in_counselor_sample=True,
                               risky=is_risk, block=block)
            shared += rng.sample(pool, 1)
    for block in BLOCKS:                              # shared risk-only cases
        pool = _sorted_ids(records, in_counselor_sample=False, in_p1=True,
                           block=block)
        shared += rng.sample(pool, 2)
    for qid in shared:
        by_id[qid]["p1_rater"] = "both"

    def remaining(**match) -> list[str]:
        return [q for q in _sorted_ids(records, **match) if q not in shared]

    for (block, is_risk), start in _COUNSELOR_DEAL_STARTS.items():
        dealt = _deal(remaining(in_counselor_sample=True, risky=is_risk,
                                block=block), rng, start)
        for qid, rater in dealt.items():
            by_id[qid]["p1_rater"] = rater
    for block, start in _RISKONLY_DEAL_STARTS.items():
        dealt = _deal(remaining(in_counselor_sample=False, in_p1=True,
                                block=block), rng, start)
        for qid, rater in dealt.items():
            by_id[qid]["p1_rater"] = rater


# Unique full-form packets drawn per (block x stratum). The 8 shared packets
# are ALWAYS full and are not drawn from here — they are what makes an
# inter-rater reliability estimate for BAGIAN 3 possible at all.
PER_STRATUM_FULL = 1


def assign_p1_forms(records: list[dict], rng: random.Random) -> None:
    """§3: set `p1_form` ("full" / "light" / None). Shared packets are always
    full-form; the rest of the full half is drawn per group and block.

    rng consumption order: counselor-risk blocks 1,2, counselor-non-risk blocks
    1,2, risk-only blocks 1,2 — `PER_STRATUM_FULL` drawn from each. Requires
    `p1_rater` to be set.

    Reduced 2026-07-30 from (2, 2, 1) per block to (1, 1, 1): 10 unique
    full-form packets -> 6, so 14 of 36 are full instead of 18. Rater workload
    drops ~30 min each. The stratification is kept — every block still
    contributes a counselor-risk, a counselor-non-risk and a risk-only packet —
    because dropping named packets instead would have skewed which kinds of case
    carry the faithfulness items.

    This function consumes the shared generator LAST (see
    freeze_eval_assignments.py), so changing these counts leaves the counselor
    sample, the rater split and the shared-8 byte-identical. That is the whole
    reason the draw order is fixed.
    """
    for r in records:
        r["p1_form"] = None

    by_id = {r["question_id"]: r for r in records}
    full: list[str] = [r["question_id"] for r in records if r["p1_rater"] == "both"]

    n = PER_STRATUM_FULL
    draws = ([({"in_counselor_sample": True, "risky": True, "block": b}, n) for b in BLOCKS]
             + [({"in_counselor_sample": True, "risky": False, "block": b}, n) for b in BLOCKS]
             + [({"in_counselor_sample": False, "in_p1": True, "block": b}, n) for b in BLOCKS])
    # Balanced by owner: WHICH case is still random, WHOSE case is not. An
    # unconstrained draw put 5 of 6 on one rater, which made that rater's load go
    # UP even though the overall count went down — the opposite of the point.
    # Shared packets already give each rater the same base, so only these need
    # balancing.
    owned = {r: 0 for r in RATERS}
    for match, k in draws:
        pool = [q for q in _sorted_ids(records, **match) if q not in full]
        for _ in range(k):
            behind = min(RATERS, key=lambda r: (owned[r], r))
            cand = [q for q in pool if by_id[q]["p1_rater"] == behind] or pool
            pick = rng.choice(cand)
            pool.remove(pick)
            full.append(pick)
            if by_id[pick]["p1_rater"] in owned:
                owned[by_id[pick]["p1_rater"]] += 1

    for r in records:
        if r["in_p1"]:
            r["p1_form"] = "full" if r["question_id"] in full else "light"


def rater_summary(records: list[dict]) -> dict:
    """Per-rater case/item counts for the freeze payload and workload checks."""
    out: dict = {}
    for rater in RATERS:
        mine = [r for r in records if r["p1_rater"] in ("both", rater)]
        counselor = [r for r in mine if r["in_counselor_sample"]]
        riskonly = [r for r in mine if not r["in_counselor_sample"]]
        out[rater] = {
            "counselor_cases": len(counselor),
            "risk_only_cases": len(riskonly),
            "p2_answers": 2 * len(counselor),
            "p1_packets": len(mine),
            "p1_full": sum(r["p1_form"] == "full" for r in mine),
            "items": 2 * len(counselor) + len(mine),
        }
    shared = [r for r in records if r["p1_rater"] == "both"]
    out["shared_items"] = sum(3 if r["in_counselor_sample"] else 1 for r in shared)
    out["p1_full_total"] = sum(r.get("p1_form") == "full" for r in records)
    return out
