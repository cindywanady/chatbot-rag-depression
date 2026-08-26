"""Segmentation: offsets, scope filter, and the criteria block."""

from depression_rag.domain import ATOMIC_CONTENT_TYPES, ContentType, SectionRole


def test_offset_invariant(segmented):
    for s in segmented.segments:
        assert segmented.cleaned_text[s.char_start : s.char_end] == s.text


def test_no_mojibake_in_cleaned_text(segmented):
    pua = {c for c in segmented.cleaned_text if 0xE000 <= ord(c) <= 0xF8FF}
    assert not pua, f"residual private-use glyphs: {sorted(hex(ord(c)) for c in pua)}"


def test_only_clinical_retained_scaffolding_dropped(segmented):
    assert all(s.section_role == SectionRole.CLINICAL for s in segmented.segments)
    roles = {s.section_role for s in segmented.dropped_segments}
    assert SectionRole.SCAFFOLDING in roles  # teaching scaffolding is dropped


def test_content_types_valid(segmented):
    valid = {c.value for c in ContentType}
    assert all(s.content_type.value in valid for s in segmented.segments)


def test_diagnostic_criteria_block_is_whole_and_atomic(segmented):
    """The 3-main + 7-additional + threshold rule must live in ONE atomic
    segment (the headline clinical-safety requirement of this pipeline)."""
    full = [
        s
        for s in segmented.segments
        if s.content_type == ContentType.CRITERIA
        and all(
            k in s.text.lower()
            for k in ("gejala utama", "gejala tambahan", "2 dari 3", "3 dari 7", "2 minggu")
        )
    ]
    assert full, "no single segment holds the complete diagnostic-criteria block"
    assert all(s.atomic for s in full)
    assert all(s.source_unit == "MI.4" for s in full)


def test_atomic_segments_are_compact(segmented, config):
    for s in segmented.segments:
        if s.atomic:
            assert s.content_type in ATOMIC_CONTENT_TYPES
            assert s.n_chars <= config.segmentation.max_atomic_chars


def test_printed_pages_within_scope_ranges(segmented, config):
    ranges = {
        u.name: (u.printed_start(config.source.printed_offset),
                 u.printed_end(config.source.printed_offset))
        for u in config.source.scoped_units()
    }
    for s in segmented.segments:
        lo, hi = ranges[s.source_unit]
        assert lo <= s.page_start <= hi
        assert lo <= s.page_end <= hi
