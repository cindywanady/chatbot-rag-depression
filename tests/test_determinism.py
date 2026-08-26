"""Reproducibility: the pipeline is deterministic end to end."""

from depression_rag.chunking import build_chunker
from depression_rag.config import load_pipeline_config
from depression_rag.domain import ChunkConfig
from depression_rag.extraction import (
    GuidelineSegmenter,
    GuidelineTextCleaner,
    PyMuPdfLoader,
)


def _segment_once(config):
    loader = PyMuPdfLoader(config.source)
    cleaner = GuidelineTextCleaner(config.cleaning)
    seg = GuidelineSegmenter(config.source, config.cleaning, config.segmentation, cleaner)
    try:
        return seg.segment(loader.load_pages(config.source.scope))
    finally:
        loader.close()


def test_segmentation_is_deterministic(config):
    a = _segment_once(config)
    b = _segment_once(config)
    assert a.cleaned_text == b.cleaned_text
    assert [s.segment_id for s in a.segments] == [s.segment_id for s in b.segments]
    assert [(s.char_start, s.char_end) for s in a.segments] == [
        (s.char_start, s.char_end) for s in b.segments
    ]


def test_chunking_is_deterministic(segmented, tokenizer):
    cfg = ChunkConfig("structure", 512, 0)
    first = build_chunker(cfg, tokenizer).chunk(segmented.cleaned_text, segmented.segments)
    second = build_chunker(cfg, tokenizer).chunk(segmented.cleaned_text, segmented.segments)
    assert [(c.chunk_id, c.char_start, c.char_end, c.text) for c in first] == [
        (c.chunk_id, c.char_start, c.char_end, c.text) for c in second
    ]
