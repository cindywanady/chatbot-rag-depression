"""Output handling: persist cleaned text, segments, and per-config chunks."""

from depression_rag.output.sink import (
    FileChunkSink,
    write_cleaned_text,
    write_segments,
)

__all__ = ["FileChunkSink", "write_cleaned_text", "write_segments"]
