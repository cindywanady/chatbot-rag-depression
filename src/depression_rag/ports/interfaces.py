"""Protocol interfaces for the pipeline stages that have more than one shape.

These are structural (``typing.Protocol``) contracts: a concrete class satisfies a
port just by having matching methods, so implementations never import each other
and the orchestrator depends only on the abstractions.

Only ports that a caller actually *names* live here. Four others once did —
``DocumentLoader``, ``Segmenter``, ``ChunkSink``, ``VectorIndex`` — each with one
implementation and zero uses of the protocol name outside its own definition. They
were removed 2026-07-26: a protocol nothing annotates against does not decouple
anything, it just adds a file to open before you can find out what a method
returns. Add one back when a second implementation appears, not before.

What survives, and why:

  * ``Chunker``     — three implementations behind ``build_chunker``
  * ``Embedder``    — two backends behind ``build_embedder``
  * ``TokenCounter``— one implementation, but the load-bearing type across
                      ``chunking/`` (8 annotation sites); the alternative is
                      importing the concrete tokenizer into every splitter
  * ``TextCleaner`` — one implementation, annotated by ``GuidelineSegmenter``,
                      which is the seam a different PDF backend would replace

Chunkers receive ``Sequence[Segment]`` rather than raw dicts: the typed domain
model keeps the contract checkable, and the plain-dict shape is still available
via :pyattr:`Segment.metadata` where serialization needs it.

Note there is no ``@runtime_checkable`` here. Nothing in the project does
``isinstance(x, SomePort)``; marking them runtime-checkable would advertise a
guarantee (that conformance is verified at run time) that is never exercised.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from depression_rag.domain import (
    Chunk,
    EmbeddingModelSpec,
    EmbeddingSide,
    PageText,
    Segment,
)


class TextCleaner(Protocol):
    """Turn a page's raw blocks into cleaned paragraph strings.

    Responsibilities: drop the running page-number header, repair line-wrap and
    hyphenation, normalize bullet / Symbol / Wingdings glyphs, preserve
    clinical glyphs and heading markers verbatim, emit valid UTF-8.
    """

    def clean_blocks(self, page: PageText) -> list[str]:
        ...


class TokenCounter(Protocol):
    """The single reference tokenizer used to size chunks across all configs.

    Fixing one tokenizer makes "256 tokens" mean the same amount of text in
    every config; each embedding model still encodes with its own tokenizer at
    indexing time.
    """

    @property
    def name(self) -> str:
        ...

    def count(self, text: str) -> int:
        ...

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Per-token ``(char_start, char_end)`` offsets into ``text``."""
        ...


class Chunker(Protocol):
    """Split cleaned text into Chunks.

    Implementations must:
      * set char offsets that index ``cleaned_text`` (the evaluation's
        relevance mapping depends on them), and
      * guarantee ``chunk.text == cleaned_text[chunk.char_start:chunk.char_end]``.
    """

    @property
    def strategy(self) -> str:
        ...

    def chunk(self, cleaned_text: str, segments: Sequence[Segment]) -> list[Chunk]:
        ...


class Embedder(Protocol):
    """Encode text into L2-normalized vectors, applying the model's prefixes.

    ``embed_queries`` and ``embed_passages`` must apply ``query_prefix`` and
    ``doc_prefix`` respectively — e5 and nomic are trained with asymmetric
    prefixes and degrade silently without them. Returned arrays are float32,
    shape ``(n, spec.dim)``, row-normalized.
    """

    @property
    def spec(self) -> EmbeddingModelSpec:
        ...

    @property
    def revision(self) -> str | None:
        ...

    @property
    def device(self) -> str:
        ...

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def n_tokens(self, texts: Sequence[str], side: EmbeddingSide | str) -> list[int]:
        """Token count per text under THIS model's own tokenizer (for truncation rate)."""
        ...

    def passage_windows(self, text: str, min_tail_tokens: int = 32) -> list[str]:
        """Split a passage into pieces that each fit ``max_seq_len`` (multi-vector
        indexing); returns ``[text]`` unchanged when it already fits."""
        ...
