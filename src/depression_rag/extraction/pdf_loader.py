"""PyMuPDF document loader - the only module that imports ``fitz``.

Extracts reading-order text *blocks* (paragraph granularity) for the in-scope
pages and records both numbering systems (physical PyMuPDF index and printed
page number). No cleaning happens here; that is the cleaner's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import fitz  # PyMuPDF

from depression_rag.config.loader import SourceConfig
from depression_rag.domain import PageText
from depression_rag.observability import get_logger

log = get_logger("pdf_loader")


class PyMuPdfLoader:
    """Loads in-scope pages as :class:`PageText`. The only place fitz/PyMuPDF is
    imported — swap this class to change the PDF backend."""

    def __init__(self, source: SourceConfig) -> None:
        self.source = source
        self._doc: fitz.Document | None = None

    def _open(self) -> fitz.Document:
        if self._doc is None:
            path = Path(self.source.pdf_path)
            if not path.is_file():
                raise FileNotFoundError(f"source PDF not found: {path}")
            self._doc = fitz.open(path)
            log.info(
                "pdf.opened",
                extra={"path": str(path), "page_count": self._doc.page_count},
            )
        return self._doc

    def load_pages(self, scope: Sequence[str]) -> list[PageText]:
        doc = self._open()
        offset = self.source.printed_offset
        pages: list[PageText] = []

        for unit_name in scope:
            unit = self.source.unit(unit_name)
            if not (1 <= unit.phys_start <= unit.phys_end <= doc.page_count):
                raise ValueError(
                    f"unit {unit_name} range {unit.phys_start}-{unit.phys_end} "
                    f"out of bounds for a {doc.page_count}-page document"
                )
            for phys in range(unit.phys_start, unit.phys_end + 1):
                page = doc[phys - 1]
                blocks = self._reading_order_blocks(page)
                pages.append(
                    PageText(
                        physical_index=phys - 1,
                        physical_page=phys,
                        printed_page=phys - offset,
                        unit=unit_name,
                        blocks=tuple(blocks),
                        n_images=len(page.get_images(full=True)),
                    )
                )
            log.info(
                "pdf.unit_loaded",
                extra={
                    "unit": unit_name,
                    "phys_range": [unit.phys_start, unit.phys_end],
                    "printed_range": [unit.phys_start - offset, unit.phys_end - offset],
                    "n_pages": unit.phys_end - unit.phys_start + 1,
                },
            )

        log.info("pdf.loaded", extra={"scope": list(scope), "n_pages": len(pages)})
        return pages

    @staticmethod
    def _reading_order_blocks(page: fitz.Page) -> list[str]:
        """Return non-empty text blocks in (top->bottom, left->right) order."""
        raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [b for b in raw if b[6] == 0]  # block_type 0 == text
        text_blocks.sort(key=lambda b: (round(b[1]), round(b[0])))
        out: list[str] = []
        for b in text_blocks:
            text = b[4].strip("\n")
            if text.strip():
                out.append(text)
        return out

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None


def rasterize_page(
    pdf_path: Path | str,
    physical_page: int,
    out_path: Path | str,
    dpi: int = 200,
) -> Path:
    """Render one *physical* (1-based) page to PNG.

    Utility for the figure-transcription workflow (inspect a raster figure
    before deciding whether to transcribe it into
    ``source.figure_transcriptions``). Kept here because it needs ``fitz``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        pix = doc[physical_page - 1].get_pixmap(dpi=dpi)
        pix.save(out_path)
    finally:
        doc.close()
    return out_path
