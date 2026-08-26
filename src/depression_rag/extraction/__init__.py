"""Extraction stage: PDF loading -> cleaning -> segmentation.

``pdf_loader`` is the *only* module that imports ``fitz`` (PyMuPDF); the cleaner
and segmenter operate purely on :class:`~depression_rag.domain.PageText`, so the
PDF backend can be swapped without touching them.
"""

from depression_rag.extraction.cleaning import GuidelineTextCleaner
from depression_rag.extraction.pdf_loader import PyMuPdfLoader, rasterize_page
from depression_rag.extraction.segmentation import GuidelineSegmenter

__all__ = [
    "GuidelineSegmenter",
    "GuidelineTextCleaner",
    "PyMuPdfLoader",
    "rasterize_page",
]
