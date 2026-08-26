"""Ports: the ``typing.Protocol`` contracts the pipeline stages speak.

Import from :mod:`depression_rag.ports.interfaces` directly — that is what every
caller in this project does. This module re-exports the same four names only so
``from depression_rag.ports import Chunker`` also works; it is not the preferred
path and nothing in the codebase uses it.

Four further ports (``DocumentLoader``, ``Segmenter``, ``ChunkSink``,
``VectorIndex``) were removed on 2026-07-26 — one implementation each, zero
annotation sites. See the ``interfaces`` module docstring for the reasoning.
"""

from depression_rag.ports.interfaces import Chunker, Embedder, TextCleaner, TokenCounter

__all__ = ["Chunker", "Embedder", "TextCleaner", "TokenCounter"]
