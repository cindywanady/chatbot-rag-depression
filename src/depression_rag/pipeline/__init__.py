"""Pipeline orchestration (composition root)."""

from depression_rag.pipeline.chunking_pipeline import ChunkingPipeline, RunResult
from depression_rag.pipeline.indexing_pipeline import IndexingPipeline, IndexRunResult

__all__ = ["ChunkingPipeline", "IndexingPipeline", "IndexRunResult", "RunResult"]
