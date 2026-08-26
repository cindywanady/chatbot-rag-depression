"""End-to-end chunking pipeline.

This is the *composition root*: the only place allowed to know the concrete
stage classes. It wires loader -> cleaner -> segmenter -> tokenizer -> chunkers
-> sink, seeds RNGs, logs every step as JSON, and writes a run manifest. Each
stage is still decoupled behind its port, so swapping one (e.g. a different PDF
backend or a custom splitter) does not touch this file's logic.
"""

from __future__ import annotations

import datetime as _dt
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from depression_rag.chunking import ReferenceTokenizer, build_chunker
from depression_rag.config.loader import PipelineConfig
from depression_rag.domain import Chunk, SegmentedDocument
from depression_rag.extraction import (
    GuidelineSegmenter,
    GuidelineTextCleaner,
    PyMuPdfLoader,
)
from depression_rag.observability import ManifestBuilder, get_logger, setup_logging
from depression_rag.output import FileChunkSink, write_cleaned_text, write_segments
from depression_rag.reproducibility import set_global_seed

log = get_logger("pipeline")


@dataclass
class RunResult:
    run_id: str
    n_pages: int
    n_segments: int
    n_dropped: int
    cleaned_chars: int
    chunks_per_config: dict[str, int] = field(default_factory=dict)
    manifest_path: str = ""
    derived_dir: str = ""
    chunks_dir: str = ""


class ChunkingPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self, run_id: str | None = None) -> RunResult:
        cfg = self.config
        run_id = run_id or _dt.datetime.now().strftime("run-%Y%m%d-%H%M%S")

        setup_logging(
            level=cfg.run.log_level,
            log_file=Path(cfg.output.logs_dir) / f"{run_id}.log",
        )
        seed_state = set_global_seed(cfg.run.seed)
        log.info("pipeline.start", extra={"run_id": run_id, "scope": list(cfg.source.scope)})

        manifest = ManifestBuilder(run_id=run_id, description="document chunking")
        manifest.add_input("source_pdf", cfg.source.pdf_path)
        manifest.add_input("config", cfg.config_path)
        # the adjudicated content_type labels are an input like any other: they
        # decide `atomic`, which decides what the structure chunker emits. Without
        # its checksum here, editing the adjudication is invisible in the manifest.
        if cfg.segmentation.content_type_overrides_path:
            manifest.add_input(
                "content_type_overrides", cfg.segmentation.content_type_overrides_path
            )
        manifest.set_config(cfg.raw)
        manifest.set_seed(seed_state)

        # --- extraction + cleaning + segmentation ---------------------------
        loader = PyMuPdfLoader(cfg.source)
        cleaner = GuidelineTextCleaner(cfg.cleaning)
        segmenter = GuidelineSegmenter(cfg.source, cfg.cleaning, cfg.segmentation, cleaner)
        try:
            pages = loader.load_pages(cfg.source.scope)
            doc = segmenter.segment(pages)
        finally:
            loader.close()

        derived = Path(cfg.output.derived_dir)
        out_paths = [write_cleaned_text(derived, doc.cleaned_text)]
        out_paths += write_segments(
            derived, doc.segments, doc.dropped_segments, cfg.output.formats
        )
        for p in out_paths:
            manifest.add_output(p)
        self._record_segment_metrics(manifest, doc, len(pages))

        # --- tokenizer (reference) -----------------------------------------
        tokenizer = ReferenceTokenizer(
            cfg.chunking.reference_tokenizer, offline=cfg.chunking.offline
        )
        manifest.set_reference_model(
            checkpoint=tokenizer.name, revision=tokenizer.revision
        )

        # --- chunking sweep --------------------------------------------------
        sink = FileChunkSink(cfg.output.chunks_dir, cfg.output.formats)
        result = RunResult(
            run_id=run_id,
            n_pages=len(pages),
            n_segments=len(doc.segments),
            n_dropped=len(doc.dropped_segments),
            cleaned_chars=doc.n_chars,
            derived_dir=str(derived),
            chunks_dir=str(cfg.output.chunks_dir),
        )

        for chunk_cfg in cfg.chunking.configs:
            t0 = time.perf_counter()
            chunker = build_chunker(chunk_cfg, tokenizer)
            chunks = chunker.chunk(doc.cleaned_text, doc.segments)
            elapsed = time.perf_counter() - t0

            self._verify_invariant(doc.cleaned_text, chunks, chunk_cfg.config_id)
            paths = sink.write_chunks(chunk_cfg.config_id, chunks)
            for p in paths:
                manifest.add_output(p, config_id=chunk_cfg.config_id)

            stats = self._chunk_stats(tokenizer, chunks, chunk_cfg.chunk_size)
            stats["build_seconds"] = round(elapsed, 3)
            manifest.record(f"config:{chunk_cfg.config_id}", stats)
            result.chunks_per_config[chunk_cfg.config_id] = len(chunks)
            log.info("pipeline.config_done", extra={"config_id": chunk_cfg.config_id, **stats})

        # written last: a manifest cannot list its own checksum, so this is the
        # end of the line — every add_output above must already have happened.
        manifest_path = manifest.write(Path(cfg.output.manifests_dir) / f"{run_id}.json")
        result.manifest_path = str(manifest_path)

        log.info(
            "pipeline.done",
            extra={
                "run_id": run_id,
                "n_segments": result.n_segments,
                "n_configs": len(cfg.chunking.configs),
                "total_chunks": sum(result.chunks_per_config.values()),
                "manifest": result.manifest_path,
            },
        )
        return result

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _record_segment_metrics(
        manifest: ManifestBuilder, doc: SegmentedDocument, n_pages: int
    ) -> None:
        manifest.record("n_pages", n_pages)
        manifest.record("n_segments_retained", len(doc.segments))
        manifest.record("n_segments_dropped", len(doc.dropped_segments))
        manifest.record("cleaned_chars", doc.n_chars)
        manifest.record("atomic_segments", sum(1 for s in doc.segments if s.atomic))
        manifest.record(
            "content_type_distribution",
            dict(Counter(s.content_type.value for s in doc.segments)),
        )
        manifest.record(
            "segments_by_unit",
            dict(Counter(s.source_unit for s in doc.segments)),
        )

    @staticmethod
    def _verify_invariant(cleaned_text: str, chunks: Sequence[Chunk], config_id: str) -> None:
        for c in chunks:
            if c.text != cleaned_text[c.char_start : c.char_end]:
                raise AssertionError(
                    f"offset invariant violated in {config_id} at chunk {c.chunk_id}"
                )

    @staticmethod
    def _chunk_stats(tokenizer: ReferenceTokenizer, chunks: Sequence[Chunk], size: int) -> dict:
        if not chunks:
            return {"n_chunks": 0}
        toks = [tokenizer.count(c.text) for c in chunks]
        atomic = sum(1 for c in chunks if c.metadata.get("atomic"))
        # chunks meaningfully over the target that are NOT atomic (diagnostic)
        oversize = sum(1 for c, t in zip(chunks, toks) if t > size + 2 and not c.metadata.get("atomic"))
        return {
            "n_chunks": len(chunks),
            "tokens_min": min(toks),
            "tokens_mean": round(sum(toks) / len(toks), 1),
            "tokens_max": max(toks),
            "atomic_chunks": atomic,
            "oversize_nonatomic": oversize,
            # per-embedding-model truncation rate is computed at indexing time
            # against each model's max_seq_len; this reports reference-tokenizer sizes.
        }
