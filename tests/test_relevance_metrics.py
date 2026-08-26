"""Unit tests for the relevance mapping + retrieval metrics (pure, no models)."""

from __future__ import annotations

import math

from depression_rag.evaluation.metrics import (
    mrr_at_k,
    multihop_coverage_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from depression_rag.evaluation.relevance import (
    RelevanceConfig,
    is_relevant,
    overlap_fraction,
    relevant_chunk_ids,
)

CFG = RelevanceConfig(denom="shorter", min_overlap_frac=0.5)


# ---- relevance mapping -----------------------------------------------------

def test_disjoint_spans_are_not_relevant():
    assert overlap_fraction((0, 100), (200, 300)) == 0.0
    assert not is_relevant((0, 100), [(200, 300)], CFG)


def test_small_gold_inside_large_chunk_is_relevant():
    # 330-char gold fully inside a 2000-char chunk: fraction over shorter == 1.0
    assert overlap_fraction((0, 2000), (500, 830), "shorter") == 1.0
    assert is_relevant((0, 2000), [(500, 830)], CFG)


def test_small_chunk_inside_large_gold_is_relevant():
    # a 400-char chunk fully inside a 24000-char gold passage
    assert overlap_fraction((1000, 1400), (0, 24000), "shorter") == 1.0
    assert is_relevant((1000, 1400), [(0, 24000)], CFG)


def test_partial_overlap_below_threshold():
    # chunk 500 wide, only 200 inside gold -> 200/500 = 0.4 < 0.5
    assert overlap_fraction((0, 500), (300, 5000), "shorter") == 0.4
    assert not is_relevant((0, 500), [(300, 5000)], CFG)


def test_denom_gold_is_spec_literal_rule():
    # 200 overlap; gold is 400 wide -> 0.5 over gold
    assert overlap_fraction((0, 200), (0, 400), "gold") == 0.5


def test_relevant_against_any_of_multiple_gold_spans():
    assert is_relevant((0, 100), [(500, 600), (50, 150)], CFG)  # 2nd span overlaps enough


def test_relevant_chunk_ids_scans_all_chunks():
    spans = {"a": (0, 100), "b": (1000, 1100), "c": (60, 160)}
    rel = relevant_chunk_ids(spans, [(50, 150)], CFG)
    assert rel == {"a", "c"}  # b is disjoint


# ---- metrics ---------------------------------------------------------------

def test_recall_at_k_is_hit_indicator():
    ranked = ["x", "y", "hit", "z"]
    assert recall_at_k(ranked, {"hit"}, 3) == 1.0
    assert recall_at_k(ranked, {"hit"}, 2) == 0.0
    assert recall_at_k(ranked, set(), 10) == 0.0


def test_precision_at_k_counts_relevant_fraction():
    ranked = ["hit1", "x", "hit2", "y"]
    assert precision_at_k(ranked, {"hit1", "hit2"}, 4) == 0.5
    assert precision_at_k(ranked, {"hit1", "hit2"}, 1) == 1.0
    assert precision_at_k(ranked, {"hit1", "hit2"}, 2) == 0.5
    assert precision_at_k(ranked, set(), 4) == 0.0
    # divisor stays k when the ranked list is shorter than k
    assert precision_at_k(["hit1"], {"hit1"}, 5) == 0.2
    assert precision_at_k(ranked, {"hit1"}, 0) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(["a", "b", "hit"], {"hit"}, 5) == 1 / 3
    assert mrr_at_k(["hit", "b"], {"hit"}, 5) == 1.0
    assert mrr_at_k(["a", "b"], {"hit"}, 5) == 0.0


def test_ndcg_perfect_and_partial():
    # single relevant at rank 1 -> perfect
    assert ndcg_at_k(["hit", "a", "b"], {"hit"}, 5) == 1.0
    # single relevant at rank 2 -> 1/log2(3) normalised by ideal 1/log2(2)=1
    got = ndcg_at_k(["a", "hit", "b"], {"hit"}, 5)
    assert math.isclose(got, 1 / math.log2(3))


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


def test_multihop_coverage_requires_all_passages():
    ranked = ["c1", "c2", "c3"]
    per_passage = [{"c1"}, {"c3"}]  # both covered in top 3
    assert multihop_coverage_at_k(ranked, per_passage, 3) == 1.0
    assert multihop_coverage_at_k(ranked, per_passage, 1) == 0.0  # c3 not in top 1
    per_passage2 = [{"c1"}, {"c99"}]  # second never retrieved
    assert multihop_coverage_at_k(ranked, per_passage2, 3) == 0.0
