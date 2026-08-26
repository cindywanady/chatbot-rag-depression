"""Build gold relevance passages from the retained segments.

A gold passage is one clinical concept with an exact char span into the cleaned
source text. They are the relevance units of the evaluation: questions are
written from them and chunk relevance is later judged by char-overlap against
their spans, identically for every chunking strategy.

The judgement layer (which segment is which concept, and where multi-concept
segments are split) lives in ``configs/gold_passages.yaml`` so it is auditable
and versioned. This module only resolves that mapping deterministically:

* whole-segment passages take the segment's ``[char_start, char_end]``;
* split passages are cut at literal ``anchor`` substrings (first occurrence
  inside the segment), each part labelled with one concept.

Every resolved span is checked against ``cleaned_text`` so the char invariant
``cleaned_text[char_start:char_end] == passage.text`` holds for all passages.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GoldPassage:
    passage_id: str
    concept: str
    source_unit: str
    source_segment_id: str
    heading_path: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    n_chars: int
    text: str

    def to_record(self) -> dict:
        return asdict(self)


@dataclass
class GoldPassageResult:
    passages: list[GoldPassage] = field(default_factory=list)
    concepts_declared: list[str] = field(default_factory=list)
    absent_concepts: dict[str, str] = field(default_factory=dict)

    @property
    def coverage(self) -> dict[str, int]:
        c = Counter(p.concept for p in self.passages)
        return {concept: c.get(concept, 0) for concept in self.concepts_declared}

    @property
    def chars_per_concept(self) -> dict[str, int]:
        out: dict[str, int] = {k: 0 for k in self.concepts_declared}
        for p in self.passages:
            out[p.concept] = out.get(p.concept, 0) + p.n_chars
        return out


_SPLIT_SUFFIXES = "abcdefghijklmnopqrstuvwxyz"


def _load_segments(path: Path) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as fh:
        segs = [json.loads(line) for line in fh if line.strip()]
    return {s["segment_id"]: s for s in segs}


def _resolve_entry(entry: dict, seg: dict, cleaned: str) -> list[GoldPassage]:
    """Turn one mapping entry into one (whole) or several (split) gold passages."""
    sid = seg["segment_id"]
    base = dict(
        source_unit=seg["source_unit"],
        source_segment_id=sid,
        heading_path=seg["heading_path"],
        page_start=seg["page_start"],
        page_end=seg["page_end"],
    )

    if "concept" in entry and "splits" not in entry:
        return [GoldPassage(passage_id=sid, concept=entry["concept"],
                            char_start=seg["char_start"], char_end=seg["char_end"],
                            n_chars=seg["n_chars"], text=seg["text"], **base)]

    # split: compute local cut points from literal anchors, then map to global offsets
    seg_text = seg["text"]
    splits = entry["splits"]
    cuts: list[int] = [0]
    for i, sp in enumerate(splits):
        if i == 0:
            if "anchor" in sp:
                raise ValueError(f"{sid}: first split must not have an anchor")
            continue
        anchor = sp["anchor"]
        local = seg_text.find(anchor)
        if local < 0:
            raise ValueError(f"{sid}: anchor not found in segment text: {anchor!r}")
        if local <= cuts[-1]:
            raise ValueError(f"{sid}: anchor {anchor!r} is out of order")
        cuts.append(local)
    cuts.append(len(seg_text))

    out: list[GoldPassage] = []
    for i, sp in enumerate(splits):
        local_start, local_end = cuts[i], cuts[i + 1]
        text = seg_text[local_start:local_end]
        # trim leading/trailing whitespace while keeping spans exact
        lstrip = len(text) - len(text.lstrip())
        rstrip = len(text) - len(text.rstrip())
        local_start += lstrip
        local_end -= rstrip
        text = seg_text[local_start:local_end]
        gstart = seg["char_start"] + local_start
        gend = seg["char_start"] + local_end
        out.append(GoldPassage(
            passage_id=f"{sid}.{_SPLIT_SUFFIXES[i]}", concept=sp["concept"],
            char_start=gstart, char_end=gend, n_chars=len(text), text=text, **base))
    return out


def build_gold_passages(config_path: str | Path, segments_path: str | Path,
                        cleaned_text_path: str | Path,
                        out_path: str | Path | None = None) -> GoldPassageResult:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    segments = _load_segments(Path(segments_path))
    cleaned = Path(cleaned_text_path).read_text(encoding="utf-8")

    declared = list(cfg["concepts"])
    result = GoldPassageResult(
        concepts_declared=declared,
        absent_concepts=dict(cfg.get("absent_concepts", {})),
    )

    for entry in cfg["passages"]:
        sid = entry["segment_id"]
        if sid not in segments:
            raise ValueError(f"segment_id not found in segments: {sid}")
        for gp in _resolve_entry(entry, segments[sid], cleaned):
            if gp.concept not in declared:
                raise ValueError(f"{gp.passage_id}: undeclared concept {gp.concept!r}")
            actual = cleaned[gp.char_start:gp.char_end]
            if actual != gp.text:
                raise ValueError(
                    f"{gp.passage_id}: char invariant violated "
                    f"(span {gp.char_start}:{gp.char_end} != stored text)")
            result.passages.append(gp)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for gp in result.passages:
                fh.write(json.dumps(gp.to_record(), ensure_ascii=False) + "\n")

    return result
