#!/usr/bin/env python
"""Build gold relevance passages and print a coverage report.

    python scripts/build_gold_passages.py

Reads configs/gold_passages.yaml + data/derived/{segments.jsonl,cleaned_text.txt}
and writes data/derived/gold_passages.jsonl.
"""

from __future__ import annotations

from pathlib import Path

from depression_rag.evaluation import build_gold_passages

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = build_gold_passages(
        config_path=ROOT / "configs" / "gold_passages.yaml",
        segments_path=ROOT / "data" / "derived" / "segments.jsonl",
        cleaned_text_path=ROOT / "data" / "derived" / "cleaned_text.txt",
        out_path=ROOT / "data" / "derived" / "gold_passages.jsonl",
    )

    n = len(result.passages)
    print(f"gold passages: {n}  (char invariant verified for all)")
    print(f"written: data/derived/gold_passages.jsonl\n")

    print("concept coverage (passages | chars):")
    chars = result.chars_per_concept
    for concept, count in result.coverage.items():
        flag = "" if count else "   <-- EMPTY"
        print(f"  {concept:24} {count:2d} | {chars[concept]:6d}{flag}")

    if result.absent_concepts:
        print("\nabsent concepts (documented, not fabricated):")
        for concept, why in result.absent_concepts.items():
            print(f"  {concept}: {why}")

    by_unit = {}
    for p in result.passages:
        by_unit[p.source_unit] = by_unit.get(p.source_unit, 0) + 1
    print("\nby source unit:", dict(sorted(by_unit.items())))


if __name__ == "__main__":
    main()
