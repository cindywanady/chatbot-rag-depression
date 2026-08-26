"""The right embedding prefix is applied per model and per side (query vs
passage). Pure - no model weights are loaded."""

import pytest

from depression_rag.domain import EmbeddingModelSpec, EmbeddingSide

# Expected (query_prefix, doc_prefix) verified from the model cards.
EXPECTED = {
    "e5-large": ("query: ", "passage: "),
    "indobert": ("", ""),
    "nomic-indonesian": ("search_query: ", "search_document: "),
    "minilm": ("", ""),
}


def test_prefix_per_model_and_side(embedding_config):
    for name, (q, d) in EXPECTED.items():
        spec = embedding_config.models[name]
        assert spec.prefix(EmbeddingSide.QUERY) == q, name
        assert spec.prefix(EmbeddingSide.PASSAGE) == d, name


def test_asymmetric_models_differ_symmetric_models_empty(embedding_config):
    for name in ("e5-large", "nomic-indonesian"):
        spec = embedding_config.models[name]
        assert spec.query_prefix and spec.doc_prefix
        assert spec.query_prefix != spec.doc_prefix
    for name in ("indobert", "minilm"):
        spec = embedding_config.models[name]
        assert spec.query_prefix == "" and spec.doc_prefix == ""


def test_apply_prefix_prepends_correct_side():
    spec = EmbeddingModelSpec(
        name="t", checkpoint="x", backend="sentence-transformers", pooling="mean",
        dim=4, query_prefix="query: ", doc_prefix="passage: ", max_seq_len=512,
    )
    assert spec.apply_prefix(["depresi", "ansietas"], EmbeddingSide.QUERY) == [
        "query: depresi", "query: ansietas",
    ]
    assert spec.apply_prefix(["isi"], EmbeddingSide.PASSAGE) == ["passage: isi"]


def test_empty_prefix_leaves_text_unchanged():
    spec = EmbeddingModelSpec(
        name="t", checkpoint="x", backend="transformers-mean", pooling="mean",
        dim=4, query_prefix="", doc_prefix="", max_seq_len=512,
    )
    assert spec.apply_prefix(["a", "b"], EmbeddingSide.QUERY) == ["a", "b"]
    assert spec.apply_prefix(["a"], EmbeddingSide.PASSAGE) == ["a"]


def test_nomic_requires_trust_remote_code(embedding_config):
    assert embedding_config.models["nomic-indonesian"].trust_remote_code is True


def test_dimensions_and_max_seq_len(embedding_config):
    m = embedding_config.models
    assert (m["e5-large"].dim, m["e5-large"].max_seq_len) == (1024, 512)
    assert (m["minilm"].dim, m["minilm"].max_seq_len) == (384, 128)
    assert m["nomic-indonesian"].max_seq_len == 8192  # per the model card
