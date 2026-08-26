"""Strategy A - fixed-size sliding window (the baseline).

A deliberately naive token window over the whole cleaned text, blind to
structure. This is the baseline the other strategies are measured against, so
it is kept structure-agnostic on purpose; each chunk's metadata is taken from
the segment it overlaps most.
"""

from __future__ import annotations

from typing import Sequence

from depression_rag.chunking.base import make_chunk, span_metadata
from depression_rag.domain import Chunk, Segment
from depression_rag.observability import get_logger
from depression_rag.ports.interfaces import TokenCounter

log = get_logger("chunking.fixed")


class FixedSizeChunker:
    """Implements the ``Chunker`` port (strategy = "fixed")."""

    def __init__(
        self,
        tokenizer: TokenCounter,
        chunk_size: int,
        overlap: int,
        params: dict,
    ) -> None:
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self._tok = tokenizer
        self._size = chunk_size
        self._overlap = overlap
        self._params = params

    @property
    def strategy(self) -> str:
        return "fixed"

    def chunk(self, cleaned_text: str, segments: Sequence[Segment]) -> list[Chunk]:
        offsets = self._tok.offsets(cleaned_text)
        n = len(offsets)
        step = self._size - self._overlap
        chunks: list[Chunk] = []
        seq = 0
        start = 0
        while start < n:
            end = min(start + self._size, n)
            window = offsets[start:end]
            cs = min(o[0] for o in window)
            ce = max(o[1] for o in window)
            if cleaned_text[cs:ce].strip():
                md = span_metadata(cs, ce, segments)
                chunks.append(
                    make_chunk(
                        "fixed", self._size, self._overlap, seq, cs, ce,
                        cleaned_text, md, self._params,
                    )
                )
                seq += 1
            if end == n:
                break
            start += step
        log.info(
            "chunking.fixed.done",
            extra={"n_chunks": len(chunks), "size": self._size, "overlap": self._overlap},
        )
        return chunks
