"""Strategy B - recursive, separator-aware splitter.

Splits on separators in priority order (paragraph -> line -> sentence -> word),
recursing into any piece still larger than the target, then packs adjacent
pieces up to the target size with token overlap. This reduces mid-sentence cuts
relative to the fixed window. A small custom implementation (rather than
LangChain's RecursiveCharacterTextSplitter) keeps the dependency surface and
the exact behaviour under our control, which matters for reproducibility.
"""

from __future__ import annotations

from typing import Sequence

from depression_rag.chunking.base import (
    DEFAULT_SEPARATORS,
    make_chunk,
    pack_spans,
    recursive_leaf_spans,
    span_metadata,
)
from depression_rag.domain import Chunk, Segment
from depression_rag.observability import get_logger
from depression_rag.ports.interfaces import TokenCounter

log = get_logger("chunking.recursive")


class RecursiveChunker:
    """Implements the ``Chunker`` port (strategy = "recursive")."""

    def __init__(
        self,
        tokenizer: TokenCounter,
        chunk_size: int,
        overlap: int,
        params: dict,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self._tok = tokenizer
        self._size = chunk_size
        self._overlap = overlap
        self._params = params
        self._separators = tuple(separators)

    @property
    def strategy(self) -> str:
        return "recursive"

    def chunk(self, cleaned_text: str, segments: Sequence[Segment]) -> list[Chunk]:
        if not cleaned_text:
            return []
        leaves = recursive_leaf_spans(
            self._tok, cleaned_text, 0, len(cleaned_text), self._size, self._separators
        )
        packed = pack_spans(self._tok, cleaned_text, leaves, self._size, self._overlap)
        chunks: list[Chunk] = []
        seq = 0
        for cs, ce in packed:
            if not cleaned_text[cs:ce].strip():
                continue
            md = span_metadata(cs, ce, segments)
            chunks.append(
                make_chunk(
                    "recursive", self._size, self._overlap, seq, cs, ce,
                    cleaned_text, md, self._params,
                )
            )
            seq += 1
        log.info(
            "chunking.recursive.done",
            extra={"n_chunks": len(chunks), "size": self._size, "overlap": self._overlap},
        )
        return chunks
