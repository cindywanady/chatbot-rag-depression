"""Text cleaning: headers, glyphs, reflow, hyphenation."""

from depression_rag.domain import PageText


def _page(blocks, printed_page=42):
    return PageText(
        physical_index=printed_page + 1,
        physical_page=printed_page + 2,
        printed_page=printed_page,
        unit="MI.4",
        blocks=tuple(blocks),
    )


def test_drops_page_number_header(cleaner):
    out = cleaner.clean_blocks(_page(["42", "Isi klinis depresi."], printed_page=42))
    assert out == ["Isi klinis depresi."]


def test_normalizes_pua_bullet(cleaner):
    out = cleaner.clean_blocks(_page([f"{chr(0xF0B7)} gejala depresi"]))
    assert chr(0xF0B7) not in out[0]
    assert out[0].startswith(cleaner.cfg.bullet_marker)


def test_line_separator_becomes_space(cleaner):
    out = cleaner.clean_blocks(_page([f"baris satu{chr(0x2028)}baris dua"]))
    assert chr(0x2028) not in out[0]
    assert out[0] == "baris satu baris dua"


def test_preserves_clinical_glyphs(cleaner):
    out = cleaner.clean_blocks(_page(["durasi ≥ 2 minggu, dosis ¼ tablet"]))
    assert "≥" in out[0]  # >=
    assert "¼" in out[0]  # 1/4


def test_dehyphenation_keeps_hyphen(cleaner):
    # Indonesian reduplication wrapped across a line must stay hyphenated
    out = cleaner.clean_blocks(_page([f"anak-{chr(10)}anak bermain"]))
    assert "anak-anak bermain" in out[0]


def test_list_items_kept_on_own_lines(cleaner):
    block = "Gejala:\n• sedih\n• lelah"
    out = cleaner.clean_blocks(_page([block]))
    assert out[0].count("\n") == 2  # heading + two bullets on separate lines
