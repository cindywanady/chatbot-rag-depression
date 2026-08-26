"""Load and validate ``configs/embedding_models.yaml`` into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from depression_rag.config.loader import RunConfig
from depression_rag.domain import EmbeddingModelSpec


@dataclass(frozen=True)
class EmbeddingRuntime:
    device: str        # "auto" | "cuda" | "cpu" (resolved at build time)
    batch_size: int
    normalize: bool


@dataclass(frozen=True)
class IndexSettings:
    type: str                          # "flat" (exact IndexFlatIP)
    metric: str                        # "ip" (cosine on normalized vectors)
    models: tuple[str, ...]            # model names to index
    chunk_configs: tuple[str, ...] | None  # None -> discover from chunks_dir
    # multi-vector mode (opt-in): chunks whose encoding exceeds a model's
    # max_seq_len are embedded as several windows, all mapping to the same
    # chunk_id (deduplicated at search). Indexes built this way get an "__mv"
    # suffix so they never overwrite the single-vector ones; leave False (or
    # omit the flag) to keep the original behaviour.
    multi_vector: bool = False


@dataclass(frozen=True)
class IndexOutputConfig:
    chunks_dir: str
    index_dir: str
    manifests_dir: str
    logs_dir: str
    formats: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingPipelineConfig:
    runtime: EmbeddingRuntime
    models: dict[str, EmbeddingModelSpec]
    index: IndexSettings
    output: IndexOutputConfig
    run: RunConfig
    config_path: str
    raw: dict[str, Any]

    def selected_models(self) -> list[EmbeddingModelSpec]:
        missing = [m for m in self.index.models if m not in self.models]
        if missing:
            raise ValueError(f"index.models refers to unknown models: {missing}")
        return [self.models[m] for m in self.index.models]


def _parse_model(name: str, m: dict) -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        name=name,
        checkpoint=m["checkpoint"],
        backend=m["backend"],
        pooling=m.get("pooling", "mean"),
        dim=int(m["dim"]),
        query_prefix=m.get("query_prefix", "") or "",
        doc_prefix=m.get("doc_prefix", "") or "",
        max_seq_len=int(m["max_seq_len"]),
        trust_remote_code=bool(m.get("trust_remote_code", False)),
    )


def load_embedding_config(path: Path | str) -> EmbeddingPipelineConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    emb = raw["embedding"]
    runtime = EmbeddingRuntime(
        device=emb.get("device", "auto"),
        batch_size=int(emb.get("batch_size", 64)),
        normalize=bool(emb.get("normalize", True)),
    )
    models = {name: _parse_model(name, spec) for name, spec in emb["models"].items()}

    for spec in models.values():
        if spec.backend not in ("sentence-transformers", "transformers-mean"):
            raise ValueError(f"unknown backend for {spec.name}: {spec.backend!r}")

    idx = raw["index"]
    chunk_configs = idx.get("chunk_configs")
    index = IndexSettings(
        type=idx.get("type", "flat"),
        metric=idx.get("metric", "ip"),
        models=tuple(idx["models"]),
        chunk_configs=tuple(chunk_configs) if chunk_configs else None,
        multi_vector=bool(idx.get("multi_vector", False)),
    )

    out = raw["output"]
    output = IndexOutputConfig(
        chunks_dir=out["chunks_dir"],
        index_dir=out["index_dir"],
        manifests_dir=out["manifests_dir"],
        logs_dir=out.get("logs_dir", "outputs/logs"),
        # jsonl is the only format anything reads back; parquet is opt-in
        formats=tuple(out.get("formats", ["jsonl"])),
    )

    run = raw.get("run", {})
    run_cfg = RunConfig(seed=int(run.get("seed", 20260628)), log_level=run.get("log_level", "INFO"))

    return EmbeddingPipelineConfig(
        runtime=runtime,
        models=models,
        index=index,
        output=output,
        run=run_cfg,
        config_path=str(path),
        raw=raw,
    )
