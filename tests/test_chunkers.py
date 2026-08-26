"""Chunkers: offset invariant, size limits, atomicity."""

import pytest

from depression_rag.chunking import build_chunker
from depression_rag.domain import ChunkConfig

SIZE_TOLERANCE = 2  # substring re-tokenization shifts boundaries by <= 2 tokens


@pytest.mark.parametrize(
    "cfg",
    [
        ChunkConfig("fixed", 128, 0),
        ChunkConfig("fixed", 256, 64),
        ChunkConfig("recursive", 256, 0),
        ChunkConfig("recursive", 512, 64),
        ChunkConfig("structure", 512, 0),
        ChunkConfig("structure", 512, 0, context_enriched=True),
    ],
)
def test_offset_invariant_and_unique_ids(cfg, segmented, tokenizer):
    chunks = build_chunker(cfg, tokenizer).chunk(segmented.cleaned_text, segmented.segments)
    assert chunks
    for c in chunks:
        assert c.text == segmented.cleaned_text[c.char_start : c.char_end]
        assert c.char_start < c.char_end
        assert c.strategy == cfg.strategy
    assert len({c.chunk_id for c in chunks}) == len(chunks)


@pytest.mark.parametrize(
    "cfg",
    [ChunkConfig("fixed", 256, 0), ChunkConfig("recursive", 256, 0), ChunkConfig("structure", 256, 0)],
)
def test_size_limits(cfg, segmented, tokenizer):
    chunks = build_chunker(cfg, tokenizer).chunk(segmented.cleaned_text, segmented.segments)
    for c in chunks:
        n = tokenizer.count(c.text)
        if cfg.strategy == "structure" and c.metadata.get("atomic"):
            continue  # atomic units stay whole even if over size (by design)
        assert n <= cfg.chunk_size + SIZE_TOLERANCE


def test_structure_keeps_criteria_block_whole(segmented, tokenizer):
    chunks = build_chunker(ChunkConfig("structure", 512, 0), tokenizer).chunk(
        segmented.cleaned_text, segmented.segments
    )
    crit = [
        s
        for s in segmented.segments
        if s.content_type.value == "criteria" and "2 minggu" in s.text.lower()
    ][0]
    # exactly one chunk equals the criteria segment span
    exact = [c for c in chunks if c.char_start == crit.char_start and c.char_end == crit.char_end]
    assert len(exact) == 1
    assert all(k in exact[0].text.lower() for k in ("gejala utama", "gejala tambahan", "2 minggu"))


def test_structure_chunks_stay_within_one_segment(segmented, tokenizer):
    chunks = build_chunker(ChunkConfig("structure", 512, 64), tokenizer).chunk(
        segmented.cleaned_text, segmented.segments
    )
    for c in chunks:
        assert c.metadata["spans_segments"] == 1


def test_fixed_chunk_id_format(segmented, tokenizer):
    chunks = build_chunker(ChunkConfig("fixed", 256, 0), tokenizer).chunk(
        segmented.cleaned_text, segmented.segments
    )
    # e.g. "fixed-256-0_MI4_0007"
    head = chunks[0].chunk_id
    assert head.startswith("fixed-256-0_")
    assert head.rsplit("_", 1)[1].isdigit()
