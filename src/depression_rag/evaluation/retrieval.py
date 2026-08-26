"""Retrieval harness for the evaluation.

Loads a built index (``outputs/indexes/<chunk_config>__<model>/``), runs the gold
questions through it with the matching embedder, and returns a ranked list of
retrieved chunk ids per question. It reuses the indexing stack — the persisted
``FaissFlatIndex`` and the same ``build_embedder`` used to build the index — so
query encoding (prefixes, pooling, normalisation) is identical to indexing.

This module only produces the *ranked ids* + *chunk spans*; scoring is done by
:mod:`relevance` (qrels) + :mod:`metrics`. Kept separate so metrics stay pure and
testable without loading models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from depression_rag.evaluation.relevance import Span

# lazy heavy imports (faiss / torch / transformers) happen inside methods


def load_chunk_spans(index_dir: str | Path) -> dict[str, Span]:
    """Map chunk_id -> (char_start, char_end) from an index's ``chunks.jsonl``."""
    path = Path(index_dir) / "chunks.jsonl"
    spans: dict[str, Span] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            spans[r["chunk_id"]] = (int(r["char_start"]), int(r["char_end"]))
    return spans


@dataclass
class RetrievalResult:
    question_id: str
    ranked_chunk_ids: list[str]
    scores: list[float]


class RetrievalRunner:
    """Query one index with its embedder. Build once per (chunk_config × model)."""

    def __init__(self, index_dir: str | Path, embedder) -> None:
        from depression_rag.indexing.faiss_index import FaissFlatIndex

        self.index_dir = Path(index_dir)
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.index = FaissFlatIndex.load(self.index_dir)
        self.chunk_spans = load_chunk_spans(self.index_dir)
        self.embedder = embedder

    def search(self, questions: list[dict], k: int) -> list[RetrievalResult]:
        """Encode every question once and search top-k (best first)."""
        texts = [q["question"] for q in questions]
        qvecs = self.embedder.embed_queries(texts)
        scores, ids = self.index.search(qvecs, k)
        return [
            RetrievalResult(question_id=q["question_id"],
                            ranked_chunk_ids=list(row_ids),
                            scores=[float(s) for s in row_scores[: len(row_ids)]])
            for q, row_ids, row_scores in zip(questions, ids, scores)
        ]


def build_embedder_for_model(model_name: str, config_path: str | Path, device: str | None = None):
    """Convenience: build the embedder for ``model_name`` from the embedding config.

    Cache the returned embedder across all chunk-config indexes of the same model
    so each model is loaded once (this is the expensive step).
    """
    from dataclasses import replace

    from depression_rag.config import load_embedding_config
    from depression_rag.embedding import build_embedder

    cfg = load_embedding_config(config_path)
    if model_name not in cfg.models:
        raise KeyError(f"unknown model {model_name!r}; known: {sorted(cfg.models)}")
    runtime = cfg.runtime if device is None else replace(cfg.runtime, device=device)
    return build_embedder(cfg.models[model_name], runtime)
