#!/usr/bin/env python
"""Reference-tokenizer sensitivity analysis (thesis robustness evidence).

Quantifies how much the choice of *reference tokenizer* (the one that defines
``chunk_size`` consistently across configs - see README) actually changes the
chunk corpus, per chunking strategy. It re-chunks the same segmented document
with the configured reference (e5/XLM-R by default) and one or more
alternatives, and reports:

  * per tokenizer: corpus token count and chars/token (the "256 tokens = a
    different amount of text" effect);
  * per (strategy x tokenizer): number of chunks;
  * per (strategy x alternative): the fraction of the *reference's* chunk
    boundaries that the alternative reproduces (within a char tolerance).

The headline finding it supports: structure-aware chunking is largely invariant
to the reference choice, while fixed/recursive corpora shift - and since the
reference is held fixed across all embedding models, it never confounds the
model comparison.

Usage (from the project root):
  ./.venv/bin/python scripts/reference_tokenizer_sensitivity.py
  ./.venv/bin/python scripts/reference_tokenizer_sensitivity.py \
      --alternatives indobenchmark/indobert-base-p1 \
                     sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
      --configs fixed-256-0 recursive-256-0 structure-512-0 \
      --out outputs/analysis/reference_tokenizer_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from depression_rag.chunking import ReferenceTokenizer, build_chunker  # noqa: E402
from depression_rag.config import load_pipeline_config  # noqa: E402
from depression_rag.domain import ChunkConfig  # noqa: E402
from depression_rag.extraction import (  # noqa: E402
    GuidelineSegmenter,
    GuidelineTextCleaner,
    PyMuPdfLoader,
)
from depression_rag.observability import setup_logging  # noqa: E402


def _parse_config(spec: str) -> ChunkConfig:
    """'fixed-256-0' or 'structure-512' -> ChunkConfig."""
    parts = spec.split("-")
    strategy = parts[0]
    size = int(parts[1])
    overlap = int(parts[2]) if len(parts) > 2 else 0
    return ChunkConfig(strategy=strategy, chunk_size=size, overlap=overlap)


def _boundaries(chunks) -> list[int]:
    """Interior chunk-end positions (the cut points), excluding end-of-corpus."""
    return sorted({c.char_end for c in chunks})[:-1]


def _reproduced(reference: list[int], other: list[int], tol: int) -> float:
    """Fraction of REFERENCE boundaries matched within ``tol`` chars in ``other``."""
    if not reference:
        return 1.0
    return sum(any(abs(x - y) <= tol for y in other) for x in reference) / len(reference)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--reference", default=None, help="baseline reference (default: config value)")
    p.add_argument(
        "--alternatives",
        nargs="+",
        default=["indobenchmark/indobert-base-p1"],
        help="alternative tokenizers to compare against the baseline",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=["fixed-256-0", "recursive-256-0", "structure-512-0"],
        help="chunk configs as strategy-size[-overlap]",
    )
    p.add_argument("--tol", type=int, default=5, help="char tolerance for boundary matching")
    p.add_argument("--offline", action="store_true", help="use cached tokenizers only")
    p.add_argument("--hf-cache", default=".hf_cache")
    p.add_argument("--out", default=None, help="write results JSON to this path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("HF_HOME", str(pathlib.Path(args.hf_cache).resolve()))
    setup_logging(level="WARNING", json_console=False)

    cfg = load_pipeline_config(args.config)

    # segment once (shared across every tokenizer)
    loader = PyMuPdfLoader(cfg.source)
    cleaner = GuidelineTextCleaner(cfg.cleaning)
    segmenter = GuidelineSegmenter(cfg.source, cfg.cleaning, cfg.segmentation, cleaner)
    try:
        doc = segmenter.segment(loader.load_pages(cfg.source.scope))
    finally:
        loader.close()

    baseline = args.reference or cfg.chunking.reference_tokenizer
    checkpoints = [baseline] + [a for a in args.alternatives if a != baseline]
    chunk_configs = [_parse_config(s) for s in args.configs]

    # load tokenizers (skip any that are unavailable rather than crashing)
    tokenizers: dict[str, ReferenceTokenizer] = {}
    tok_stats: dict[str, dict] = {}
    for ckpt in checkpoints:
        try:
            tok = ReferenceTokenizer(ckpt, offline=args.offline)
        except Exception as exc:  # network/cache miss
            print(f"  ! skipping {ckpt}: {type(exc).__name__}")
            continue
        tokenizers[ckpt] = tok
        n = tok.count(doc.cleaned_text)
        tok_stats[ckpt] = {
            "revision": tok.revision,
            "corpus_tokens": n,
            "chars_per_token": round(doc.n_chars / n, 3),
        }

    if baseline not in tokenizers:
        raise SystemExit(f"baseline reference {baseline!r} could not be loaded")

    # per-config chunking + boundary agreement vs the baseline
    results: dict[str, dict] = {}
    for cc in chunk_configs:
        ref_chunks = build_chunker(cc, tokenizers[baseline]).chunk(doc.cleaned_text, doc.segments)
        ref_bounds = _boundaries(ref_chunks)
        per_tok: dict[str, dict] = {}
        for ckpt, tok in tokenizers.items():
            chunks = build_chunker(cc, tok).chunk(doc.cleaned_text, doc.segments)
            per_tok[ckpt] = {
                "n_chunks": len(chunks),
                "ref_boundaries_reproduced": (
                    None if ckpt == baseline
                    else round(_reproduced(ref_bounds, _boundaries(chunks), args.tol), 3)
                ),
            }
        results[cc.config_id] = per_tok

    report = {
        "config": args.config,
        "scope": list(cfg.source.scope),
        "char_tolerance": args.tol,
        "corpus_chars": doc.n_chars,
        "reference": baseline,
        "tokenizers": tok_stats,
        "results": results,
    }

    # pretty print
    print(f"\ncorpus: {doc.n_chars} chars | reference: {baseline}")
    print("\nper-tokenizer corpus size:")
    for ckpt, s in tok_stats.items():
        tag = "  (REFERENCE)" if ckpt == baseline else ""
        print(f"  {ckpt:55} {s['corpus_tokens']:>7} tok  {s['chars_per_token']:>5} ch/tok{tag}")
    print(f"\nboundary agreement vs reference (within {args.tol} chars):")
    header = f"  {'config':16} " + " ".join(f"{c.split('/')[-1][:22]:>22}" for c in tokenizers)
    print(header)
    for cid, per_tok in results.items():
        cells = []
        for ckpt in tokenizers:
            r = per_tok[ckpt]
            val = "ref" if r["ref_boundaries_reproduced"] is None else f"{r['ref_boundaries_reproduced']:.0%}"
            cells.append(f"{val + ' (' + str(r['n_chunks']) + ')':>22}")
        print(f"  {cid:16} " + " ".join(cells))
    print("  (cell = boundary agreement vs reference; n_chunks in parentheses)")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
