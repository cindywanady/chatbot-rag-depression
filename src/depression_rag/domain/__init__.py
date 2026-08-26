"""Domain models: the stable vocabulary shared across all pipeline stages.

These are plain, mostly-frozen dataclasses with no I/O and no third-party
dependencies, so every other module can depend on them without creating
coupling between stages.
"""

from depression_rag.domain.models import (
    ATOMIC_CONTENT_TYPES,
    SEGMENT_SEPARATOR,
    Chunk,
    ChunkConfig,
    ContentType,
    EmbeddingModelSpec,
    EmbeddingSide,
    PageText,
    SectionRole,
    Segment,
    SegmentedDocument,
    SourceUnit,
)

__all__ = [
    "ATOMIC_CONTENT_TYPES",
    "SEGMENT_SEPARATOR",
    "Chunk",
    "ChunkConfig",
    "ContentType",
    "EmbeddingModelSpec",
    "EmbeddingSide",
    "PageText",
    "SectionRole",
    "Segment",
    "SegmentedDocument",
    "SourceUnit",
]
