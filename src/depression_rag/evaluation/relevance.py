"""Cross-strategy relevance mapping.

Relevance is defined at the SOURCE-SPAN level and projected onto each strategy's
chunks by char-span overlap, so "relevant" means the same thing for every
index regardless of how it was chunked. This module is pure and deterministic:
given a chunk's ``[char_start, char_end]`` and the gold passage span(s) a question
cites, it decides relevance under the single frozen threshold from
``configs/relevance.yaml``.

The char offsets are the shared coordinate system: gold passages and every chunk
carry offsets into the same ``cleaned_text.txt`` (domain invariant
``text == cleaned_text[char_start:char_end]``), so overlap is exact, not fuzzy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

Span = tuple[int, int]  # (char_start, char_end), end exclusive


@dataclass(frozen=True)
class RelevanceConfig:
    """The frozen relevance threshold (declared once, then never tuned)."""

    denom: str = "shorter"          # "shorter" | "gold" | "chunk"
    min_overlap_frac: float = 0.5
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    primary_metric: str = "ndcg@5"
    graded: bool = False
    # which file this came from — reports name it rather than claiming the metric
    # was "declared" somewhere unspecified (see pipeline_audit.md §4, correction 1)
    source_path: str = ""
    # models present in the grid at selection time; post-selection challengers are
    # scored but must not silently decide a "the selection changed" claim
    selection_grid_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.denom not in ("shorter", "gold", "chunk"):
            raise ValueError(f"denom must be shorter|gold|chunk, got {self.denom!r}")
        if not 0.0 < self.min_overlap_frac <= 1.0:
            raise ValueError(f"min_overlap_frac must be in (0, 1], got {self.min_overlap_frac}")


def load_relevance_config(path: str | Path) -> RelevanceConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["relevance"]
    return RelevanceConfig(
        denom=raw.get("denom", "shorter"),
        min_overlap_frac=float(raw.get("min_overlap_frac", 0.5)),
        k_values=tuple(int(k) for k in raw.get("k_values", (1, 3, 5, 10))),
        primary_metric=str(raw.get("primary_metric", "ndcg@5")),
        graded=bool(raw.get("graded", False)),
        source_path=str(path),
        selection_grid_models=tuple(raw.get("selection_grid_models", []) or []),
    )


def overlap_chars(a: Span, b: Span) -> int:
    """Length of the char intersection of two spans (0 if disjoint)."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def overlap_fraction(chunk: Span, gold: Span, denom: str = "shorter") -> float:
    """Overlap of ``chunk`` and ``gold`` normalised by the chosen denominator span.

    ``shorter`` (default) makes containment either direction score 1.0, which is
    what makes the rule robust to the gold-passage size range (see relevance.yaml).
    """
    inter = overlap_chars(chunk, gold)
    if inter == 0:
        return 0.0
    clen, glen = chunk[1] - chunk[0], gold[1] - gold[0]
    if denom == "gold":
        base = glen
    elif denom == "chunk":
        base = clen
    else:  # shorter
        base = min(clen, glen)
    return inter / base if base > 0 else 0.0


def is_relevant(chunk: Span, gold_spans: list[Span], cfg: RelevanceConfig) -> bool:
    """A chunk is relevant if it clears the threshold for ANY cited gold span.

    (In scoring, a hit against any one of a multi-hop question's passages counts;
    all-passage coverage is checked separately by ``multihop_coverage_at_k``.)
    """
    return any(
        overlap_fraction(chunk, g, cfg.denom) >= cfg.min_overlap_frac
        for g in gold_spans
    )


def relevant_chunk_ids(
    chunk_spans: dict[str, Span], gold_spans: list[Span], cfg: RelevanceConfig
) -> set[str]:
    """All chunk ids in one index that are relevant to a question (its qrels).

    Scanning every chunk (not just the retrieved ones) gives the total relevant
    count that Recall / nDCG need for their denominator / ideal ranking.
    """
    return {cid for cid, span in chunk_spans.items() if is_relevant(span, gold_spans, cfg)}


def gold_spans_for(passage_ids: list[str], gold_by_id: dict[str, Span]) -> list[Span]:
    """Resolve a question's ``passage_ids`` to their gold char spans."""
    missing = [p for p in passage_ids if p not in gold_by_id]
    if missing:
        raise KeyError(f"question cites unknown gold passage_ids: {missing}")
    return [gold_by_id[p] for p in passage_ids]
