"""Raw-transformers backend with explicit mean pooling (IndoBERT).

IndoBERT is a masked-LM, not a sentence-embedding model, so we pool explicitly
rather than relying on sentence-transformers' auto-wrapping. This is the
intentional weak baseline (Reimers & Gurevych, 2019). Heavy imports are lazy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from depression_rag.domain import EmbeddingModelSpec, EmbeddingSide
from depression_rag.embedding._hf import cache_revision, window_spans
from depression_rag.observability import get_logger

log = get_logger("embedding.transformers_mean")


class TransformersMeanEmbedder:
    """Implements the ``Embedder`` port via transformers + manual mean pooling."""

    def __init__(
        self,
        spec: EmbeddingModelSpec,
        device: str,
        batch_size: int,
        normalize: bool,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._spec = spec
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint)
        self._model = (
            AutoModel.from_pretrained(spec.checkpoint, trust_remote_code=spec.trust_remote_code)
            .to(device)
            .eval()
        )
        log.info(
            "embedder.loaded",
            extra={"model": spec.name, "checkpoint": spec.checkpoint,
                   "device": device, "dim": spec.dim, "max_seq_len": spec.max_seq_len,
                   "backend": "transformers-mean"},
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

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: Sequence[str], side: EmbeddingSide) -> np.ndarray:
        torch = self._torch
        prefixed = self._spec.apply_prefix(list(texts), side)
        out: list[np.ndarray] = []
        for i in range(0, len(prefixed), self._batch_size):
            batch = prefixed[i : i + self._batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._spec.max_seq_len,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state
            emb = self._mean_pool(hidden, enc["attention_mask"])
            if self._normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
        if not out:
            return np.zeros((0, self._spec.dim), dtype="float32")
        return np.ascontiguousarray(np.vstack(out), dtype="float32")

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, EmbeddingSide.PASSAGE)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, EmbeddingSide.QUERY)

    def n_tokens(self, texts: Sequence[str], side: EmbeddingSide | str) -> list[int]:
        prefixed = self._spec.apply_prefix(list(texts), EmbeddingSide(side))
        return [
            len(self._tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"])
            for t in prefixed
        ]

    def passage_windows(self, text: str, min_tail_tokens: int = 32) -> list[str]:
        """Split ``text`` into pieces that each fit ``max_seq_len`` when encoded
        as a passage. Returns ``[text]`` when it already fits (see the
        sentence-transformers backend for the multi-vector rationale).
        """
        tok = self._tokenizer
        if not getattr(tok, "is_fast", False):
            return [text]
        prefix = self._spec.doc_prefix
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        budget = self._spec.max_seq_len - n_prefix - tok.num_special_tokens_to_add()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= budget:
            return [text]
        spans = window_spans(enc["offset_mapping"], budget, min_tail_tokens)
        return [text[s:e] for s, e in spans]
