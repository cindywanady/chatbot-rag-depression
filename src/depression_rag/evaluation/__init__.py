"""Retrieval evaluation: gold passages, question set, relevance mapping, metrics."""

from depression_rag.evaluation.gold_passages import (
    GoldPassage,
    GoldPassageResult,
    build_gold_passages,
)
from depression_rag.evaluation.questions import (
    QuestionValidation,
    load_questions,
    validate_questions,
)
from depression_rag.evaluation.relevance import (
    RelevanceConfig,
    gold_spans_for,
    is_relevant,
    load_relevance_config,
    overlap_fraction,
    relevant_chunk_ids,
)
from depression_rag.evaluation.metrics import (
    QueryScore,
    mean_scores,
    mrr_at_k,
    multihop_coverage_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_query,
)
from depression_rag.evaluation.bootstrap import (
    CI,
    PairedTest,
    bootstrap_ci,
    holm_adjust,
    paired_bootstrap_diff,
    rank_biserial,
)

__all__ = [
    "GoldPassage",
    "GoldPassageResult",
    "build_gold_passages",
    "QuestionValidation",
    "load_questions",
    "validate_questions",
    # relevance mapping
    "RelevanceConfig",
    "load_relevance_config",
    "overlap_fraction",
    "is_relevant",
    "relevant_chunk_ids",
    "gold_spans_for",
    # metrics
    "QueryScore",
    "score_query",
    "mean_scores",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "multihop_coverage_at_k",
    # bootstrap CI + significance
    "CI",
    "PairedTest",
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "holm_adjust",
    "rank_biserial",
]
