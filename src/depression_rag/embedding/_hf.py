"""Small Hugging Face helpers shared by the embedders."""

from __future__ import annotations

import os
from pathlib import Path


def window_spans(
    offsets: list[tuple[int, int]],
    window_tokens: int,
    min_tail_tokens: int = 32,
) -> list[tuple[int, int]]:
    """Consecutive char spans of ~``window_tokens`` tokens each (multi-vector mode).

    ``offsets`` are per-token ``(char_start, char_end)`` offsets into a text.
    A final partial window shorter than ``min_tail_tokens`` is dropped: losing
    a couple of dozen tokens is the same trivial regime as single-vector
    truncation, and a near-empty window would add an uninformative vector.
    """
    n = len(offsets)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = min(i + window_tokens, n)
        if spans and j - i < min_tail_tokens:
            break
        window = offsets[i:j]
        spans.append((min(o[0] for o in window), max(o[1] for o in window)))
        i = j
    return spans


def cache_revision(checkpoint: str) -> str | None:
    """Resolve a model's commit hash from the in-project HF cache (for manifests)."""
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None
    ref = (
        Path(hf_home)
        / "hub"
        / f"models--{checkpoint.replace('/', '--')}"
        / "refs"
        / "main"
    )
    if ref.is_file():
        return ref.read_text(encoding="utf-8").strip()
    return None
