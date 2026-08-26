"""Build a chunker from a :class:`ChunkConfig` (factory keeps callers decoupled)."""

from __future__ import annotations

from depression_rag.chunking.fixed import FixedSizeChunker
from depression_rag.chunking.recursive import RecursiveChunker
from depression_rag.chunking.structure import StructureAwareChunker
from depression_rag.domain import ChunkConfig
from depression_rag.ports.interfaces import Chunker, TokenCounter


def build_chunker(config: ChunkConfig, tokenizer: TokenCounter) -> Chunker:
    """Return the concrete :class:`Chunker` for ``config.strategy``."""
    if config.strategy == "fixed":
        return FixedSizeChunker(
            tokenizer, config.chunk_size, config.overlap, config.params
        )
    if config.strategy == "recursive":
        return RecursiveChunker(
            tokenizer, config.chunk_size, config.overlap, config.params
        )
    if config.strategy == "structure":
        return StructureAwareChunker(
            tokenizer,
            config.chunk_size,
            config.overlap,
            config.context_enriched,
            config.params,
        )
    raise ValueError(f"unknown chunking strategy: {config.strategy!r}")
