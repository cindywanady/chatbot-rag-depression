"""Rank-aware retrieval metrics.

All metrics take a per-query ranked list of retrieved chunk ids (best first) and
the set of relevant chunk ids for that query (its qrels, from
:mod:`~depression_rag.evaluation.relevance`). Relevance is binary here (every gold
passage graded 1); nDCG is written so graded gains can drop in later.

Definitions:
  * Recall@k: is a relevant chunk in the top k — a success/hit rate.
  * Precision@k: fraction of the top k that is relevant.
  * MRR@k: reciprocal rank of the FIRST relevant chunk (0 if none in top k).
  * nDCG@k (primary, per configs/relevance.yaml): position-aware, DCG
    normalised by the ideal DCG.
Plus a multi-hop coverage check: does the top k contain a relevant chunk for
EVERY passage the question requires, not just one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Recall@k as used in this evaluation: 1.0 if any relevant chunk is in the top k.

    (A per-query hit indicator averaged over queries — i.e. hit-rate@k — not the
    |hits|/|relevant| ratio. The name matches the label used in the results
    tables; the distinction is deliberate and documented there.)
    """
    return 1.0 if any(c in relevant for c in ranked[:k]) else 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Precision@k: |top-k ∩ relevant| / k.

    The divisor is k even if fewer than k chunks were retrieved (standard IR
    convention; never happens here — every index holds > 10 chunks). Note that
    with overlapping chunking configs the relevant set contains near-duplicate
    chunks, which inflates precision for those configs (the same redundancy
    caveat applies to recall@1 and MRR), so do not compare precision across
    configs with different chunk granularity.
    """
    if k <= 0:
        return 0.0
    return sum(1 for c in ranked[:k] if c in relevant) / k


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k. IDCG places all relevant chunks at the top."""
    if not relevant:
        return 0.0
    gains = [1.0 if c in relevant else 0.0 for c in ranked]
    dcg = dcg_at_k(gains, k)
    ideal = dcg_at_k([1.0] * len(relevant), k)
    return dcg / ideal if ideal > 0 else 0.0


def multihop_coverage_at_k(
    ranked: list[str], relevant_per_passage: list[set[str]], k: int
) -> float:
    """1.0 only if top-k contains ≥1 relevant chunk for EVERY required passage.

    ``relevant_per_passage[i]`` is the qrels for the i-th cited gold passage alone.
    For single-passage questions this reduces to recall_at_k.
    """
    top = set(ranked[:k])
    return 1.0 if all(top & rel for rel in relevant_per_passage) else 0.0


@dataclass
class QueryScore:
    """Per-query metric values, keyed ``"<metric>@<k>"`` (e.g. ``"ndcg@5"``)."""

    question_id: str
    question_type: str
    difficulty: str
    n_relevant: int
    values: dict[str, float]


def score_query(
    question_id: str,
    question_type: str,
    difficulty: str,
    ranked: list[str],
    relevant: set[str],
    k_values: tuple[int, ...],
    relevant_per_passage: list[set[str]] | None = None,
) -> QueryScore:
    values: dict[str, float] = {}
    for k in k_values:
        values[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        values[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        values[f"mrr@{k}"] = mrr_at_k(ranked, relevant, k)
        values[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
        if relevant_per_passage is not None:
            values[f"multihop_coverage@{k}"] = multihop_coverage_at_k(
                ranked, relevant_per_passage, k
            )
    return QueryScore(
        question_id=question_id,
        question_type=question_type,
        difficulty=difficulty,
        n_relevant=len(relevant),
        values=values,
    )


def mean_scores(scores: list[QueryScore]) -> dict[str, float]:
    """Macro-average each metric over queries (empty -> {})."""
    if not scores:
        return {}
    keys = scores[0].values.keys()
    n = len(scores)
    return {key: sum(s.values.get(key, 0.0) for s in scores) / n for key in keys}
