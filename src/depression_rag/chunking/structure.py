"""Strategy C - structure-aware chunker.

Splits on the document's own hierarchy: the segmenter already produced one
segment per Pokok Bahasan / subsection, so the unit of chunking is the segment.

Two hard rules:
  1. Never split a tightly-coupled clinical unit. Segments flagged ``atomic``
     (the diagnostic-criteria block, dose blocks, referral lists, the somatic
     table) are emitted whole as a single chunk even if that exceeds the
     nominal size.
  2. Optionally prepend the heading_path at EMBED time (the "context-enriched"
     variant). Chunk text is not modified here - the offset invariant must
     hold - we only record ``context_enriched`` in params and keep heading_path
     in metadata, so the indexing stage applies it uniformly.

Non-atomic segments larger than the target are split with the same recursive,
separator-aware splitter used by Strategy B, so their pieces respect sentence
and paragraph boundaries.
"""

from __future__ import annotations

from typing import Sequence

from depression_rag.chunking.base import (
    DEFAULT_SEPARATORS,
    make_chunk,
    pack_spans,
    recursive_leaf_spans,
)
from depression_rag.domain import Chunk, Segment
from depression_rag.observability import get_logger
from depression_rag.ports.interfaces import TokenCounter

log = get_logger("chunking.structure")


class StructureAwareChunker:
    """Implements the ``Chunker`` port (strategy = "structure")."""

    def __init__(
        self,
        tokenizer: TokenCounter,
        chunk_size: int,
        overlap: int,
        context_enriched: bool,
        params: dict,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self._tok = tokenizer
        self._size = chunk_size
        self._overlap = overlap
        self._context_enriched = context_enriched
        self._params = params
        self._separators = tuple(separators)

    @property
    def strategy(self) -> str:
        return "structure"

    def chunk(self, cleaned_text: str, segments: Sequence[Segment]) -> list[Chunk]:
        chunks: list[Chunk] = []
        seq = 0
        kept_atomic = 0
        for seg in segments:
            spans = self._segment_spans(cleaned_text, seg)
            if seg.atomic and len(spans) == 1:
                kept_atomic += 1
            for cs, ce in spans:
                if not cleaned_text[cs:ce].strip():
                    continue
                md = dict(seg.metadata)
                md["spans_segments"] = 1  # a structure chunk lives inside one segment
                chunks.append(
                    make_chunk(
                        "structure", self._size, self._overlap, seq, cs, ce,
                        cleaned_text, md, self._params,
                    )
                )
                seq += 1
        log.info(
            "chunking.structure.done",
            extra={
                "n_chunks": len(chunks),
                "size": self._size,
                "overlap": self._overlap,
                "context_enriched": self._context_enriched,
                "atomic_kept_whole": kept_atomic,
            },
        )
        return chunks

    def _segment_spans(self, cleaned_text: str, seg: Segment) -> list[tuple[int, int]]:
        # Rule 1: atomic units and already-small segments stay whole.
        if seg.atomic or self._tok.count(seg.text) <= self._size:
            return [(seg.char_start, seg.char_end)]
        # Otherwise split this segment with the recursive splitter, then pack.
        leaves = recursive_leaf_spans(
            self._tok, cleaned_text, seg.char_start, seg.char_end, self._size, self._separators
        )
        return pack_spans(self._tok, cleaned_text, leaves, self._size, self._overlap)
