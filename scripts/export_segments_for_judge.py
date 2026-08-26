#!/usr/bin/env python
"""Export a BLIND input for the content_type LLM-judge audit.

Writes data/derived/segments_for_judge.jsonl with only segment_id, source_unit,
heading_path, n_chars, text — deliberately WITHOUT content_type/atomic, so the
judge classifies independently (no anchoring on the rule label). Attach the
output file to the judge prompt (configs/prompts/content_type_audit.md).

    python scripts/export_segments_for_judge.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ("segment_id", "source_unit", "heading_path", "n_chars", "text")


def main() -> None:
    src = ROOT / "data" / "derived" / "segments.jsonl"
    out = ROOT / "data" / "derived" / "segments_for_judge.jsonl"
    n = 0
    with open(src, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            s = json.loads(line)
            fout.write(json.dumps({k: s[k] for k in KEEP}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} blind segments -> data/derived/segments_for_judge.jsonl "
          f"(fields: {', '.join(KEEP)}; content_type withheld)")


if __name__ == "__main__":
    main()
