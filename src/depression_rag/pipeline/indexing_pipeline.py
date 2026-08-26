"""Embedding + indexing pipeline.

For every (chunk config x embedding model) it loads the persisted chunks, embeds
the passages with that model (applying its doc prefix, L2-normalized), builds an
exact FAISS IndexFlatIP, and persists the index + chunk metadata + a per-index
meta record (checkpoint, revision, device, n_chunks, mean/max tokens, truncation
rate, build time). Each model is loaded once and reused across all chunk configs.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from depression_rag.config.embedding_config import EmbeddingPipelineConfig
from depression_rag.domain import EmbeddingModelSpec, EmbeddingSide
from depression_rag.embedding import build_embedder, resolve_device
from depression_rag.indexing import build_flat_index
from depression_rag.observability import ManifestBuilder, get_logger, setup_logging
from depression_rag.ports.interfaces import Embedder
from depression_rag.reproducibility import set_global_seed

log = get_logger("indexing_pipeline")


@dataclass
class IndexRunResult:
    run_id: str
    n_indexes: int = 0
    index_ids: list[str] = field(default_factory=list)
    manifest_path: str = ""
    index_dir: str = ""


def _load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _discover_config_ids(chunks_dir: Path) -> list[str]:
    return sorted(p.stem for p in chunks_dir.glob("*.jsonl"))


def _encoder_inputs(chunks: list[dict]) -> list[str]:
    """Text actually fed to the embedder.

    For context-enriched configs, prepend the heading_path so the passage
    vector carries section context. The stored chunk text and its
    char offsets are unchanged - only the encoder input differs, and only the
    document side (queries are never enriched).
    """
    inputs: list[str] = []
    for c in chunks:
        text = c["text"]
        if c.get("param_context_enriched") and c.get("meta_heading_path"):
            text = f"{c['meta_heading_path']}\n{text}"
        inputs.append(text)
    return inputs


class IndexingPipeline:
    def __init__(self, config: EmbeddingPipelineConfig) -> None:
        self.config = config

    def run(self, run_id: str | None = None) -> IndexRunResult:
        cfg = self.config
        run_id = run_id or _dt.datetime.now().strftime("index-%Y%m%d-%H%M%S")
        setup_logging(level=cfg.run.log_level, log_file=Path(cfg.output.logs_dir) / f"{run_id}.log")
        seed_state = set_global_seed(cfg.run.seed)
        device = resolve_device(cfg.runtime.device)

        chunks_dir = Path(cfg.output.chunks_dir)
        config_ids = list(cfg.index.chunk_configs) if cfg.index.chunk_configs else _discover_config_ids(chunks_dir)
        if not config_ids:
            raise SystemExit(f"no chunk files found in {chunks_dir} (run the chunking pipeline first)")
        specs = cfg.selected_models()

        log.info("indexing.start", extra={"run_id": run_id, "device": device,
                                          "n_chunk_configs": len(config_ids), "models": [s.name for s in specs],
                                          "n_indexes_planned": len(config_ids) * len(specs)})

        manifest = ManifestBuilder(run_id=run_id, description="embedding + indexing")
        manifest.add_input("embedding_config", cfg.config_path)
        # The chunk files ARE the content of every index built here. Without their
        # checksums a manifest cannot answer "which chunk set is this index?" — and
        # the chunking stage rewrites those files in place, so the question is real.
        for config_id in config_ids:
            manifest.add_input(f"chunks:{config_id}", chunks_dir / f"{config_id}.jsonl")
        manifest.set_config(cfg.raw)
        manifest.set_seed(seed_state)
        manifest.record("device", device)

        result = IndexRunResult(run_id=run_id, index_dir=str(cfg.output.index_dir))

        for spec in specs:
            embedder = build_embedder(spec, cfg.runtime)  # load model once
            manifest.record(
                f"model:{spec.name}",
                {"checkpoint": spec.checkpoint, "revision": embedder.revision,
                 "device": embedder.device, "dim": spec.dim, "max_seq_len": spec.max_seq_len,
                 "backend": spec.backend},
            )
            for config_id in config_ids:
                self._build_one(cfg, spec, embedder, chunks_dir, config_id, manifest, result, device)

        manifest_path = manifest.write(Path(cfg.output.manifests_dir) / f"{run_id}.json")
        result.manifest_path = str(manifest_path)
        result.n_indexes = len(result.index_ids)
        log.info("indexing.done", extra={"run_id": run_id, "n_indexes": result.n_indexes,
                                         "manifest": result.manifest_path})
        return result

    def _build_one(self, cfg, spec: EmbeddingModelSpec, embedder: Embedder, chunks_dir: Path,
                   config_id: str, manifest: ManifestBuilder, result: IndexRunResult, device: str) -> None:
        chunks = _load_chunks(chunks_dir / f"{config_id}.jsonl")
        if not chunks:
            log.warning("indexing.empty_config", extra={"config_id": config_id})
            return
        texts = _encoder_inputs(chunks)  # heading_path prepended iff context_enriched
        ids = [c["chunk_id"] for c in chunks]

        tok_counts = embedder.n_tokens(texts, EmbeddingSide.PASSAGE)
        truncated = sum(1 for n in tok_counts if n > spec.max_seq_len)

        multi_vector = cfg.index.multi_vector
        if multi_vector:
            # one vector per <= max_seq_len window; every window keeps its
            # chunk's id and search collapses duplicates (FaissFlatIndex)
            enc_texts, enc_ids, windowed = [], [], 0
            for text, cid in zip(texts, ids):
                wins = embedder.passage_windows(text)
                windowed += len(wins) > 1
                enc_texts.extend(wins)
                enc_ids.extend([cid] * len(wins))
        else:
            enc_texts, enc_ids, windowed = texts, ids, 0

        t0 = time.perf_counter()
        vectors = embedder.embed_passages(enc_texts)  # (n_vectors, dim), normalized
        index = build_flat_index(vectors, enc_ids, metric=cfg.index.metric)
        build_seconds = round(time.perf_counter() - t0, 3)

        # multi-vector indexes get their own directory so they never replace
        # the single-vector ones; delete the __mv dir to retire the variant
        index_id = f"{config_id}__{spec.name}" + ("__mv" if multi_vector else "")
        out_dir = Path(cfg.output.index_dir) / index_id
        written = index.save(out_dir)
        written += self._write_chunk_table(out_dir, chunks, cfg.output.formats)

        meta = {
            "index_id": index_id,
            "chunk_config_id": config_id,
            "model": spec.name,
            "checkpoint": spec.checkpoint,
            "revision": embedder.revision,
            "device": device,
            "dim": spec.dim,
            "metric": cfg.index.metric,
            "normalized": cfg.runtime.normalize,
            "context_enriched": bool(chunks[0].get("param_context_enriched")),
            "multi_vector": multi_vector,
            "n_chunks": len(chunks),
            "n_vectors": len(enc_texts),
            "windowed_chunks": windowed,
            "tokens_mean": round(sum(tok_counts) / len(tok_counts), 1),
            "tokens_max": max(tok_counts),
            "max_seq_len": spec.max_seq_len,
            # single-vector view of the chunk/cap interaction; with
            # multi_vector these chunks are windowed instead of clipped
            "truncated_chunks": truncated,
            "truncation_rate": round(truncated / len(chunks), 4),
            "build_seconds": build_seconds,
        }
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(meta_path)

        # every file this index dir contains, not just the vectors: ids.json maps
        # faiss positions to chunk_ids and chunks.jsonl carries the char spans the
        # relevance mapping needs, so an index with a valid .faiss and a stale
        # chunks.jsonl scores wrong while looking intact
        for path in written:
            manifest.add_output(path, index_id=index_id)
        manifest.record(f"index:{index_id}", meta)
        result.index_ids.append(index_id)
        log.info("indexing.index_built", extra=meta)

    @staticmethod
    def _write_chunk_table(out_dir: Path, chunks: list[dict], formats) -> list[Path]:
        written: list[Path] = []
        if "jsonl" in formats:
            p = out_dir / "chunks.jsonl"
            with open(p, "w", encoding="utf-8") as fh:
                for c in chunks:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            written.append(p)
        if "parquet" in formats:
            import pandas as pd

            p = out_dir / "chunks.parquet"
            pd.DataFrame.from_records(chunks).to_parquet(p, index=False)
            written.append(p)
        return written
