"""Command-line entry point for the chunking pipeline.

Run from the project root (paths in the config are relative to it)::

    python -m depression_rag.cli                       # full default run
    python -m depression_rag.cli --scope MI.4          # MI.4 only
    python -m depression_rag.cli --only fixed-256-0 structure-512-0
    depression-rag-chunk --config configs/pipeline.yaml

By default the Hugging Face cache is kept INSIDE the project (./.hf_cache) so no
data leaves chatbot_dep.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from depression_rag.config import load_pipeline_config
from depression_rag.config.loader import PipelineConfig


def _banner(title: str, lines: list[str]) -> None:
    """Loud stderr notice for a run that writes partial results to shared paths."""
    width = 78
    print("\n" + "!" * width, file=sys.stderr)
    print(f"!! {title}", file=sys.stderr)
    for line in lines:
        print(f"!! {line}", file=sys.stderr)
    print("!" * width + "\n", file=sys.stderr)


def _apply_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    source = cfg.source
    if args.scope:
        unknown = [s for s in args.scope if s not in {u.name for u in source.units}]
        if unknown:
            raise SystemExit(f"--scope refers to unknown units: {unknown}")
        if tuple(args.scope) != source.scope:
            # cleaned_text.txt IS the coordinate system: gold passages, every
            # chunk and every index store char offsets into it. Rewriting it from
            # a narrower scope does not fail — it silently renumbers everything.
            _banner(
                "PARTIAL SCOPE — shared artifacts will be OVERWRITTEN in place",
                [f"scope: {', '.join(source.scope)}  ->  {', '.join(args.scope)}",
                 "",
                 "About to rewrite the char-offset coordinate system:",
                 f"  {cfg.output.derived_dir}/cleaned_text.txt",
                 f"  {cfg.output.derived_dir}/segments.jsonl",
                 f"  {cfg.output.chunks_dir}/*.jsonl",
                 "",
                 "gold_passages.jsonl, the built indexes and every retrieval score",
                 "hold offsets into the FULL-scope text and will no longer line up.",
                 "To explore a subset safely, redirect output.derived_dir and",
                 "output.chunks_dir in a copy of the config first.",
                 "Recover the frozen set with: python -m depression_rag.cli"],
            )
        source = replace(source, scope=tuple(args.scope))

    output = cfg.output
    if args.formats:
        output = replace(output, formats=tuple(args.formats))

    chunking = cfg.chunking
    if args.only:
        wanted = set(args.only)
        selected = tuple(c for c in chunking.configs if c.config_id in wanted)
        missing = wanted - {c.config_id for c in selected}
        if missing:
            raise SystemExit(f"--only refers to unknown config_ids: {sorted(missing)}")
        if len(selected) < len(chunking.configs):
            _banner(
                "PARTIAL CONFIG SET — the chunks directory will hold a MIX of runs",
                [f"running {len(selected)} of {len(chunking.configs)} configs: "
                 f"{', '.join(c.config_id for c in selected)}",
                 "",
                 f"The other {len(chunking.configs) - len(selected)} chunk files stay on disk from an earlier run,",
                 "and nothing on the file records which run produced them.",
                 "Re-run without --only before indexing or evaluating."],
            )
        chunking = replace(chunking, configs=selected)

    run = cfg.run
    if args.log_level:
        run = replace(run, log_level=args.log_level)

    return replace(cfg, source=source, output=output, chunking=chunking, run=run)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="depression-rag-chunk", description=__doc__)
    p.add_argument("--config", default="configs/pipeline.yaml", help="pipeline YAML")
    p.add_argument("--run-id", default=None, help="run id (default: timestamp)")
    p.add_argument("--scope", nargs="+", metavar="MI.x", help="override MI units to index")
    p.add_argument("--only", nargs="+", metavar="CONFIG_ID", help="run only these chunk configs")
    p.add_argument("--formats", nargs="+", choices=["jsonl", "parquet"], help="output formats")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    p.add_argument("--hf-cache", default=".hf_cache", help="Hugging Face cache dir (kept in-project)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # keep model/tokenizer downloads inside the project
    os.environ.setdefault("HF_HOME", str(Path(args.hf_cache).resolve()))

    cfg = _apply_overrides(load_pipeline_config(args.config), args)

    # imported here so HF_HOME is set before transformers loads
    from depression_rag.pipeline import ChunkingPipeline

    result = ChunkingPipeline(cfg).run(run_id=args.run_id)

    print(
        f"\nDone: run_id={result.run_id}\n"
        f"  segments retained: {result.n_segments} (dropped {result.n_dropped}) "
        f"from {result.n_pages} pages, {result.cleaned_chars} cleaned chars\n"
        f"  configs: {len(result.chunks_per_config)} | "
        f"total chunks: {sum(result.chunks_per_config.values())}\n"
        f"  chunks dir: {result.chunks_dir}\n"
        f"  manifest:   {result.manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
