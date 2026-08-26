"""Command-line entry point for the embedding + indexing pipeline.

Run from the project root, after the chunking pipeline has produced
``outputs/chunks/``:

    python -m depression_rag.cli_index                       # all configs x all models
    python -m depression_rag.cli_index --models minilm --configs structure-512-0
    python -m depression_rag.cli_index --device cpu
    depression-rag-index --config configs/embedding_models.yaml

The Hugging Face cache stays inside the project (./.hf_cache).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from depression_rag.config import load_embedding_config
from depression_rag.config.embedding_config import EmbeddingPipelineConfig


def _apply_overrides(cfg: EmbeddingPipelineConfig, args: argparse.Namespace) -> EmbeddingPipelineConfig:
    index = cfg.index
    if args.models:
        unknown = [m for m in args.models if m not in cfg.models]
        if unknown:
            raise SystemExit(f"--models refers to unknown models: {unknown}")
        index = replace(index, models=tuple(args.models))
    if args.configs:
        # match --models: name a config that has no chunk file and say so here,
        # rather than surfacing a bare FileNotFoundError from deep in the run
        available = {p.stem for p in Path(cfg.output.chunks_dir).glob("*.jsonl")}
        unknown = [c for c in args.configs if c not in available]
        if unknown:
            raise SystemExit(
                f"--configs refers to chunk configs with no file in "
                f"{cfg.output.chunks_dir}: {unknown}\n"
                f"available: {sorted(available)}\n"
                "(run the chunking pipeline first if the config is new)"
            )
        index = replace(index, chunk_configs=tuple(args.configs))

    if args.multi_vector:
        index = replace(index, multi_vector=True)

    runtime = cfg.runtime
    if args.device:
        runtime = replace(runtime, device=args.device)
    if args.batch_size:
        runtime = replace(runtime, batch_size=args.batch_size)

    run = cfg.run
    if args.log_level:
        run = replace(run, log_level=args.log_level)

    return replace(cfg, index=index, runtime=runtime, run=run)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="depression-rag-index", description=__doc__)
    p.add_argument("--config", default="configs/embedding_models.yaml")
    p.add_argument("--run-id", default=None)
    p.add_argument("--models", nargs="+", help="override which embedding models to run")
    p.add_argument("--configs", nargs="+", metavar="CONFIG_ID", help="override which chunk configs to index")
    p.add_argument("--device", default=None, help="auto|cuda|cpu")
    p.add_argument("--multi-vector", action="store_true",
                   help="embed oversize chunks as multiple windows per chunk_id; "
                        "writes to separate '<config>__<model>__mv' index dirs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--log-level", default=None)
    p.add_argument("--hf-cache", default=".hf_cache")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("HF_HOME", str(Path(args.hf_cache).resolve()))

    cfg = _apply_overrides(load_embedding_config(args.config), args)

    from depression_rag.pipeline import IndexingPipeline

    result = IndexingPipeline(cfg).run(run_id=args.run_id)
    print(
        f"\nDone: run_id={result.run_id}\n"
        f"  indexes built: {result.n_indexes}\n"
        f"  index dir:     {result.index_dir}\n"
        f"  manifest:      {result.manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
