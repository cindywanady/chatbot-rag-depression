"""Text cleaning.

Turns a page's raw blocks into cleaned paragraph strings:

  * drop the running page-number header (the printed number is the first block);
  * normalize Symbol/Wingdings PUA bullet glyphs to one marker, while
    PRESERVING clinical glyphs (>=, down-arrow, 1/4, 1/2, en-dash) and the
    heading markers (MI.x, Pokok Bahasan X, roman/numbered) verbatim;
  * repair line-wrap (reflow soft-wrapped lines into paragraphs) and
    hyphenation (join ``word-\\nword`` keeping the hyphen, which is safe for
    Indonesian reduplication like ``anak-anak``);
  * keep list items on their own lines so the segmenter and chunkers can see
    list structure.

Pure text in / text out - no PDF dependency.
"""

from __future__ import annotations

import re
import unicodedata

from depression_rag.config.loader import CleaningConfig
from depression_rag.domain import PageText

# Characters normalized unconditionally, independent of the YAML glyph map.
# Keys use chr(codepoint) so the source stays pure-ASCII and unambiguous
# (these characters are invisible and do not survive copy/paste reliably).
_UNCONDITIONAL = {
    chr(0x2028): "\n",  # LINE SEPARATOR -> newline
    chr(0x2029): "\n",  # PARAGRAPH SEPARATOR -> newline
    chr(0x00A0): " ",   # NO-BREAK SPACE -> space
    chr(0x00AD): "",    # SOFT HYPHEN -> drop
    chr(0xFEFF): "",    # ZERO WIDTH NO-BREAK SPACE / BOM -> drop
}

# A line that starts a list item / enumerated clause - keep it on its own line.
_LIST_START = re.compile(
    r"^(?:[•‣◦▪\-–—*]|→|✓"
    r"|\d+[.)]\s|[a-zA-Z][.)]\s|[ivxIVX]+[.)]\s)"
)
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")


class GuidelineTextCleaner:
    """Implements the ``TextCleaner`` port for this guideline."""

    def __init__(self, cleaning: CleaningConfig) -> None:
        self.cfg = cleaning
        # one translation table: unconditional rules + configured glyph map
        table: dict[int, str] = {ord(k): v for k, v in _UNCONDITIONAL.items()}
        for char, replacement in cleaning.glyph_map.items():
            table[ord(char)] = replacement
        self._translation = table

    def clean_blocks(self, page: PageText) -> list[str]:
        """Return cleaned, non-empty paragraph blocks for one page."""
        cleaned: list[str] = []
        for block in page.blocks:
            text = self._clean_block(block, page)
            if text:
                cleaned.append(text)
        return cleaned

    # -- internals ----------------------------------------------------------
    def _clean_block(self, block: str, page: PageText) -> str:
        if self.cfg.drop_page_number_header and self._is_page_number(block, page):
            return ""
        text = block.translate(self._translation)
        text = unicodedata.normalize("NFC", text)
        text = self._reflow(text)
        text = _MULTISPACE.sub(" ", text)
        text = _MULTINEWLINE.sub("\n\n", text)
        return text.strip()

    def _is_page_number(self, block: str, page: PageText) -> bool:
        s = block.strip()
        if page.printed_page is not None and s == str(page.printed_page):
            return True
        # fall back to a bare 1-3 digit block (the running header position)
        return bool(_PAGE_NUMBER.match(s))

    def _reflow(self, text: str) -> str:
        """Join soft-wrapped lines into paragraphs; keep list items separate."""
        lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            return ""
        out = lines[0]
        for line in lines[1:]:
            if _LIST_START.match(line):
                out += "\n" + line
            elif self.cfg.dehyphenate and out.endswith("-"):
                # de-hyphenation: keep the hyphen, drop the line break
                out += line
            else:
                out += " " + line
        return out
