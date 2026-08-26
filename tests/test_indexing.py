"""FAISS IndexFlatIP wrapper + multi-vector windowing - synthetic, no models."""

import numpy as np
import pytest

from depression_rag.embedding._hf import window_spans
from depression_rag.indexing import FaissFlatIndex, build_flat_index


def _unit_rows(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_self_retrieval_top1_is_identity():
    v = _unit_rows(20, 16)
    ids = [f"chunk-{i:02d}" for i in range(20)]
    idx = build_flat_index(v, ids)
    assert idx.size == 20 and idx.dim == 16
    scores, rids = idx.search(v, k=1)
    assert [r[0] for r in rids] == ids                  # each vector retrieves itself
    assert np.allclose(scores[:, 0], 1.0, atol=1e-4)    # cosine self-similarity == 1


def test_ranking_is_by_cosine():
    # a clear nearest neighbour should come back first
    base = _unit_rows(5, 8, seed=1)
    idx = build_flat_index(base, [f"c{i}" for i in range(5)])
    _, rids = idx.search(base[2:3], k=2)
    assert rids[0][0] == "c2"


def test_save_and_load_roundtrip(tmp_path):
    v = _unit_rows(12, 8)
    ids = [f"c{i}" for i in range(12)]
    build_flat_index(v, ids).save(tmp_path)
    assert (tmp_path / "index.faiss").exists() and (tmp_path / "ids.json").exists()
    reloaded = FaissFlatIndex.load(tmp_path)
    assert reloaded.size == 12 and reloaded.dim == 8
    s1, r1 = reloaded.search(v, k=1)
    assert [r[0] for r in r1] == ids
    assert np.allclose(s1[:, 0], 1.0, atol=1e-4)


def test_dim_mismatch_raises():
    idx = FaissFlatIndex(8)
    with pytest.raises(ValueError):
        idx.add(_unit_rows(3, 16), ["a", "b", "c"])


def test_id_count_must_match_vectors():
    idx = FaissFlatIndex(8)
    with pytest.raises(ValueError):
        idx.add(_unit_rows(3, 8), ["only-one-id"])


def test_only_inner_product_metric_supported():
    with pytest.raises(ValueError):
        FaissFlatIndex(8, metric="l2")


# ---- multi-vector: duplicate-id search + windowing --------------------------

def test_duplicate_ids_are_deduplicated_keeping_best_window():
    # chunk "big" has two window vectors; a query near window 2 must return
    # "big" once, scored by its best window, with unique ids filling top-k
    v = _unit_rows(4, 8, seed=3)
    idx = build_flat_index(v, ["big", "big", "c1", "c2"])
    scores, rids = idx.search(v[1:2], k=3)  # query == big's 2nd window
    assert rids[0][0] == "big"
    assert len(rids[0]) == len(set(rids[0])) == 3      # unique, still k ids
    assert np.isclose(scores[0][0], 1.0, atol=1e-4)     # best window's score

    # single-vector indexes keep the plain path (identical behaviour)
    plain = build_flat_index(v, ["a", "b", "c", "d"])
    s2, r2 = plain.search(v[1:2], k=3)
    assert r2[0][0] == "b" and np.isclose(s2[0][0], 1.0, atol=1e-4)


def test_duplicate_ids_roundtrip_through_save_load(tmp_path):
    v = _unit_rows(3, 8, seed=4)
    build_flat_index(v, ["x", "x", "y"]).save(tmp_path)
    reloaded = FaissFlatIndex.load(tmp_path)
    _, rids = reloaded.search(v[0:1], k=2)
    assert rids[0] == ["x", "y"]                        # dedup survives reload


def test_window_spans_covers_text_and_drops_tiny_tail():
    # 10 tokens, 1 char each at positions 0..9
    offs = [(i, i + 1) for i in range(10)]
    assert window_spans(offs, window_tokens=4, min_tail_tokens=1) == [
        (0, 4), (4, 8), (8, 10)]
    # tail of 2 tokens < min_tail 3 -> dropped
    assert window_spans(offs, window_tokens=4, min_tail_tokens=3) == [(0, 4), (4, 8)]
    # fits in one window -> single span
    assert window_spans(offs, window_tokens=32) == [(0, 10)]
    assert window_spans([], 4) == []
