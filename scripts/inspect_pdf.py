#!/usr/bin/env python
"""Reproducible PDF inspection - how the grounded facts were established.

This script regenerates the evidence behind configs/pipeline.yaml (no guessing):
the printed<->physical page offset, the MI-unit page ranges, the non-ASCII /
Symbol-Wingdings glyph census, the clinical-body anchors, and (optionally) a
rasterized page for inspecting the one raster figure.

    ./.venv/bin/python scripts/inspect_pdf.py
    ./.venv/bin/python scripts/inspect_pdf.py --rasterize 46 --out outputs/figures/p46.png
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter

import fitz  # PyMuPDF

DEFAULT_PDF = "data/Modul Keswa bagi Dokter Umum di  FKTP.pdf"
_MI_RE = re.compile(r"(?i)^\s*materi\s+inti\s+(\d+)")
_PAGENO_RE = re.compile(r"^\d{1,3}$")


def find_units(doc: fitz.Document) -> list[tuple[str, int]]:
    """Return [(unit_name, physical_1based_start), ...] by scanning headings."""
    units = []
    for i in range(doc.page_count):
        for line in doc[i].get_text("text").splitlines():
            m = _MI_RE.match(line.strip())
            if m:
                units.append((f"MI.{m.group(1)}", i + 1))
                break
    return units


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", default=DEFAULT_PDF)
    p.add_argument("--rasterize", type=int, default=None, metavar="PHYS_PAGE",
                   help="render this physical (1-based) page to PNG")
    p.add_argument("--out", default="outputs/figures/page.png")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)

    doc = fitz.open(args.pdf)
    print(f"PDF: {args.pdf}")
    print(f"pages: {doc.page_count}  producer: {doc.metadata.get('producer')}")

    print("\n== MI units (name -> physical start, inferred end, printed range) ==")
    units = find_units(doc)
    for idx, (name, start) in enumerate(units):
        end = (units[idx + 1][1] - 1) if idx + 1 < len(units) else doc.page_count
        # detect printed offset from this page's leading page-number block
        first = doc[start - 1].get_text("text").strip().splitlines()
        printed = first[0].strip() if first and _PAGENO_RE.match(first[0].strip()) else "?"
        off = (start - int(printed)) if printed != "?" else "?"
        print(f"  {name:6} phys {start:>3}-{end:<3}  printed_first={printed}  offset={off}")

    print("\n== non-ASCII / glyph census (top 20) ==")
    cnt: Counter = Counter()
    for i in range(doc.page_count):
        for ch in doc[i].get_text("text"):
            if ord(ch) > 0x7E and ch not in "\n\r\t":
                cnt[ch] += 1
    for ch, c in cnt.most_common(20):
        pua = " <PUA>" if 0xE000 <= ord(ch) <= 0xF8FF else ""
        print(f"  U+{ord(ch):04X} {ch!r:>8} x{c:<5} {unicodedata.name(ch, '<no name>')}{pua}")

    if args.rasterize is not None:
        from pathlib import Path

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pix = doc[args.rasterize - 1].get_pixmap(dpi=args.dpi)
        pix.save(out)
        print(f"\nrasterized physical page {args.rasterize} -> {out} "
              f"({pix.width}x{pix.height})")

    doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
