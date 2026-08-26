"""sentence-transformers backend (e5, nomic, MiniLM).

Heavy imports are lazy so importing the package does not pull in torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from depression_rag.domain import EmbeddingModelSpec, EmbeddingSide
from depression_rag.embedding._hf import cache_revision, window_spans
from depression_rag.observability import get_logger

log = get_logger("embedding.sentence_transformers")


class SentenceTransformerEmbedder:
    """Implements the ``Embedder`` port via sentence-transformers."""

    def __init__(
        self,
        spec: EmbeddingModelSpec,
        device: str,
        batch_size: int,
        normalize: bool,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._spec = spec
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = SentenceTransformer(
            spec.checkpoint,
            device=device,
            trust_remote_code=spec.trust_remote_code,
        )
        # enforce the model card's truncation point as the sequence limit
        self._model.max_seq_length = spec.max_seq_len
        log.info(
            "embedder.loaded",
            extra={"model": spec.name, "checkpoint": spec.checkpoint,
                   "device": device, "dim": spec.dim, "max_seq_len": spec.max_seq_len},
        )

    @property
    def spec(self) -> EmbeddingModelSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    @property
    def revision(self) -> str | None:
        return cache_revision(self._spec.checkpoint)

    def _encode(self, texts: Sequence[str], side: EmbeddingSide) -> np.ndarray:
        prefixed = self._spec.apply_prefix(list(texts), side)
        emb = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(emb, dtype="float32")

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, EmbeddingSide.PASSAGE)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, EmbeddingSide.QUERY)

    def n_tokens(self, texts: Sequence[str], side: EmbeddingSide | str) -> list[int]:
        """Untruncated token length per text (to compute the truncation rate)."""
        prefixed = self._spec.apply_prefix(list(texts), EmbeddingSide(side))
        tok = self._model.tokenizer
        return [len(tok(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in prefixed]

    def passage_windows(self, text: str, min_tail_tokens: int = 32) -> list[str]:
        """Split ``text`` into pieces that each fit ``max_seq_len`` when encoded
        as a passage (doc prefix + special tokens included). Returns ``[text]``
        unchanged when it already fits — the multi-vector opt-in hook.
        """
        tok = self._model.tokenizer
        if not getattr(tok, "is_fast", False):  # offsets need a fast tokenizer
            return [text]
        prefix = self._spec.doc_prefix
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        budget = self._spec.max_seq_len - n_prefix - tok.num_special_tokens_to_add()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= budget:
            return [text]
        spans = window_spans(enc["offset_mapping"], budget, min_tail_tokens)
        return [text[s:e] for s, e in spans]
