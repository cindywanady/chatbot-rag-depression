"""Build an Embedder from a spec + runtime (keeps callers decoupled)."""

from __future__ import annotations

from depression_rag.config.embedding_config import EmbeddingRuntime
from depression_rag.domain import EmbeddingModelSpec
from depression_rag.ports.interfaces import Embedder


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda' if a GPU is visible, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_embedder(spec: EmbeddingModelSpec, runtime: EmbeddingRuntime) -> Embedder:
    device = resolve_device(runtime.device)
    if spec.backend == "sentence-transformers":
        from depression_rag.embedding.sentence_transformers_embedder import (
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder(spec, device, runtime.batch_size, runtime.normalize)
    if spec.backend == "transformers-mean":
        from depression_rag.embedding.transformers_mean_embedder import (
            TransformersMeanEmbedder,
        )

        return TransformersMeanEmbedder(spec, device, runtime.batch_size, runtime.normalize)
    raise ValueError(f"unknown embedding backend: {spec.backend!r}")
