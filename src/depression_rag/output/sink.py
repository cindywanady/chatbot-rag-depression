"""Persisting pipeline outputs as JSONL and/or Parquet.

  * the canonical ``cleaned_text.txt`` (the offset coordinate system),
  * ``segments.jsonl`` (retained) + ``segments_dropped.jsonl`` (audit),
  * one chunk file per ``config_id``.

JSONL is the always-on, human-inspectable, diff-able format, and it is what every
consumer in this repo reads: the indexing stage, the retrieval harness and the
chatbot all open ``*.jsonl``.

``_write_parquet`` is kept because ``output.formats`` can still ask for it, but
the project no longer does (dropped 2026-07-26): nothing here ever read a parquet
back, so the copies were pure duplication. Re-enable it only for columnar
analysis *outside* the pipeline — and do not let a reader start depending on it
without noticing that the writer is optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from depression_rag.domain import Chunk, Segment
from depression_rag.observability import get_logger

log = get_logger("output")


def _write_jsonl(path: Path, records: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
    return path


def _write_parquet(path: Path, records: Sequence[dict]) -> Path:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(list(records)).to_parquet(path, index=False)
    return path


def _segment_record(seg: Segment) -> dict:
    return {
        "segment_id": seg.segment_id,
        "source_unit": seg.source_unit,
        "pokok_bahasan": seg.pokok_bahasan,
        "heading_path": seg.heading_path,
        "page_start": seg.page_start,
        "page_end": seg.page_end,
        "content_type": seg.content_type.value,
        "section_role": seg.section_role.value,
        "atomic": seg.atomic,
        "lang": seg.lang,
        "char_start": seg.char_start,
        "char_end": seg.char_end,
        "n_chars": seg.n_chars,
        "text": seg.text,
    }


def write_cleaned_text(derived_dir: Path | str, cleaned_text: str) -> Path:
    path = Path(derived_dir) / "cleaned_text.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cleaned_text, encoding="utf-8")
    log.info("output.cleaned_text", extra={"path": str(path), "chars": len(cleaned_text)})
    return path


def write_segments(
    derived_dir: Path | str,
    segments: Sequence[Segment],
    dropped: Sequence[Segment],
    formats: Sequence[str] = ("jsonl",),
) -> list[Path]:
    derived = Path(derived_dir)
    written: list[Path] = []
    kept_records = [_segment_record(s) for s in segments]
    drop_records = [_segment_record(s) for s in dropped]

    written.append(_write_jsonl(derived / "segments.jsonl", kept_records))
    written.append(_write_jsonl(derived / "segments_dropped.jsonl", drop_records))
    if "parquet" in formats and kept_records:
        written.append(_write_parquet(derived / "segments.parquet", kept_records))

    log.info(
        "output.segments",
        extra={"retained": len(kept_records), "dropped": len(drop_records),
               "files": [str(p) for p in written]},
    )
    return written


class FileChunkSink:
    """Writes one chunk file per ``config_id``; returns the paths so the run
    manifest can checksum them."""

    def __init__(self, chunks_dir: Path | str, formats: Sequence[str] = ("jsonl",)) -> None:
        self.chunks_dir = Path(chunks_dir)
        self.formats = tuple(formats)

    def write_chunks(self, config_id: str, chunks: Sequence[Chunk]) -> list[Path]:
        records = [c.to_record() for c in chunks]
        written: list[Path] = []
        if "jsonl" in self.formats or not self.formats:
            written.append(_write_jsonl(self.chunks_dir / f"{config_id}.jsonl", records))
        if "parquet" in self.formats and records:
            written.append(_write_parquet(self.chunks_dir / f"{config_id}.parquet", records))
        log.info(
            "output.chunks",
            extra={"config_id": config_id, "n_chunks": len(records),
                   "files": [str(p) for p in written]},
        )
        return written
