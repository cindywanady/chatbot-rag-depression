"""Shared chunking primitives.

Everything works in CHAR offsets into ``cleaned_text`` and guarantees that a
returned span ``(s, e)`` satisfies ``chunk.text == cleaned_text[s:e]``. Token
counts are obtained from the reference tokenizer so all strategies measure size
the same way.
"""

from __future__ import annotations

from typing import Sequence

from depression_rag.domain import Chunk, Segment
from depression_rag.ports.interfaces import TokenCounter

# Recursive separators in priority order (paragraph -> line -> sentence -> word).
# The final "" triggers a hard token-window split for unsplittable spans.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def dominant_segment(
    char_start: int, char_end: int, segments: Sequence[Segment]
) -> tuple[Segment | None, int]:
    """Return (segment with the largest char overlap, #segments overlapped)."""
    best: Segment | None = None
    best_overlap = 0
    n_overlap = 0
    for seg in segments:
        overlap = min(char_end, seg.char_end) - max(char_start, seg.char_start)
        if overlap > 0:
            n_overlap += 1
            if overlap > best_overlap:
                best_overlap = overlap
                best = seg
    return best, n_overlap


def span_metadata(
    char_start: int, char_end: int, segments: Sequence[Segment]
) -> dict:
    """Metadata for a span = the dominant segment's metadata + span diagnostics."""
    seg, n = dominant_segment(char_start, char_end, segments)
    if seg is None:  # should not happen for in-bounds spans
        md: dict = {
            "source_unit": None,
            "pokok_bahasan": None,
            "heading_path": None,
            "page_start": None,
            "page_end": None,
            "content_type": None,
            "lang": "id",
            "segment_id": None,
            "atomic": False,
        }
    else:
        md = dict(seg.metadata)
    md["spans_segments"] = n
    return md


def _unit_token(source_unit: str | None) -> str:
    return (source_unit or "NA").replace(".", "").replace(" ", "")


def make_chunk(
    strategy: str,
    chunk_size: int,
    overlap: int,
    seq: int,
    char_start: int,
    char_end: int,
    cleaned_text: str,
    metadata: dict,
    params: dict,
) -> Chunk:
    """Build a Chunk, deriving ``text`` by slicing so the offset invariant holds."""
    text = cleaned_text[char_start:char_end]
    # chunk_id = <strategy>-<size>-<overlap>_<UNIT>_<seq>.
    #
    # WARNING - the trailing number is a GLOBAL counter over the whole document
    # (0..n-1, never restarting per unit); the unit token is only a readability tag
    # copied from the chunk's dominant segment. It is NOT a segment id, but it has
    # exactly the same shape as one, so the two id families collide: on this corpus
    # 47 of the 90 structure-512-0 chunk ids end in a string that IS a real
    # segment id, and almost always a different one -
    #     structure-512-0_MI1_0002  comes from segment  MI1_0001
    # (only ..._MI1_0000 happens to coincide). Never recover provenance by parsing a
    # chunk_id: read metadata["segment_id"] (serialised as meta_segment_id).
    chunk_id = (
        f"{strategy}-{chunk_size}-{overlap}"
        f"_{_unit_token(metadata.get('source_unit'))}_{seq:04d}"
    )
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        char_start=char_start,
        char_end=char_end,
        metadata=metadata,
        strategy=strategy,
        params=params,
    )


def _split_on_separator(
    text: str, start: int, end: int, sep: str
) -> list[tuple[int, int]]:
    """Partition ``[start, end)`` into contiguous spans, cutting AFTER each ``sep``.

    Spans are contiguous and cover the whole range (the separator stays attached
    to the preceding piece), so concatenating them reproduces the text exactly.
    """
    spans: list[tuple[int, int]] = []
    cursor = start
    pos = text.find(sep, cursor, end)
    while pos != -1:
        cut = pos + len(sep)
        if cut >= end:
            break
        spans.append((cursor, cut))
        cursor = cut
        pos = text.find(sep, cursor, end)
    if cursor < end:
        spans.append((cursor, end))
    return spans


def token_window_spans(
    tokenizer: TokenCounter, text: str, start: int, end: int, size: int
) -> list[tuple[int, int]]:
    """Hard fallback: contiguous <= size-token spans over ``[start, end)``."""
    sub = text[start:end]
    offs = tokenizer.offsets(sub)
    if not offs:
        return [(start, end)]
    spans: list[tuple[int, int]] = []
    i, n = 0, len(offs)
    while i < n:
        j = min(i + size, n)
        window = offs[i:j]
        s = start + min(o[0] for o in window)
        e = start + max(o[1] for o in window)
        spans.append((s, e))
        i = j
    return spans


def recursive_leaf_spans(
    tokenizer: TokenCounter,
    text: str,
    start: int,
    end: int,
    size: int,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[tuple[int, int]]:
    """Recursively split ``[start, end)`` into leaf spans each <= ``size`` tokens."""
    if tokenizer.count(text[start:end]) <= size:
        return [(start, end)]
    for k, sep in enumerate(separators):
        if sep == "":
            return token_window_spans(tokenizer, text, start, end, size)
        parts = _split_on_separator(text, start, end, sep)
        if len(parts) > 1:
            leaves: list[tuple[int, int]] = []
            for ps, pe in parts:
                if tokenizer.count(text[ps:pe]) <= size:
                    leaves.append((ps, pe))
                else:
                    leaves.extend(
                        recursive_leaf_spans(
                            tokenizer, text, ps, pe, size, separators[k + 1 :]
                        )
                    )
            return leaves
    return token_window_spans(tokenizer, text, start, end, size)


def pack_spans(
    tokenizer: TokenCounter,
    text: str,
    leaves: Sequence[tuple[int, int]],
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Greedily pack leaf spans into <= ``size``-token chunks with token overlap."""
    if not leaves:
        return []
    counts = [tokenizer.count(text[s:e]) for s, e in leaves]
    chunks: list[tuple[int, int]] = []
    i, n = 0, len(leaves)
    while i < n:
        j, total = i, 0
        while j < n:
            if total + counts[j] > size and j > i:
                break
            total += counts[j]
            j += 1
        chunks.append((leaves[i][0], leaves[j - 1][1]))
        if j >= n:
            break
        if overlap > 0:  # step back to re-include ~overlap tokens of context
            back, k = 0, j - 1
            while k > i and back < overlap:
                back += counts[k]
                k -= 1
            i = max(k + 1, i + 1)  # always make progress
        else:
            i = j
    return chunks
