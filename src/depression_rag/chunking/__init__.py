"""Chunking stage: three strategies behind the ``Chunker`` port.

  * :class:`FixedSizeChunker`      - token sliding window (naive baseline)
  * :class:`RecursiveChunker`      - separator-aware split (paragraph->sentence->word)
  * :class:`StructureAwareChunker` - split on the document hierarchy, never
                                     splitting a tightly-coupled clinical unit

All three measure "tokens" with one shared :class:`ReferenceTokenizer` so a
"256-token" chunk means the same amount of text across configs, and all
guarantee ``chunk.text == cleaned_text[chunk.char_start:chunk.char_end]``.
"""

from depression_rag.chunking.fixed import FixedSizeChunker
from depression_rag.chunking.recursive import RecursiveChunker
from depression_rag.chunking.registry import build_chunker
from depression_rag.chunking.structure import StructureAwareChunker
from depression_rag.chunking.tokenizer import ReferenceTokenizer

__all__ = [
    "FixedSizeChunker",
    "RecursiveChunker",
    "ReferenceTokenizer",
    "StructureAwareChunker",
    "build_chunker",
]
