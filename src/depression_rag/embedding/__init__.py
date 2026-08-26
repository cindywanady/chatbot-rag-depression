"""Embedding stage: encode chunks/queries into normalized vectors.

Three of the four models use the sentence-transformers backend; IndoBERT uses a
raw-transformers backend with explicit mean pooling (it is not a sentence
embedding model - the intentional weak baseline). All apply the model's
asymmetric query/passage prefixes at encode time.
"""

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from depression_rag.embedding.registry import build_embedder, resolve_device
from depression_rag.embedding.sentence_transformers_embedder import (
    SentenceTransformerEmbedder,
)
from depression_rag.embedding.transformers_mean_embedder import TransformersMeanEmbedder

__all__ = [
    "SentenceTransformerEmbedder",
    "TransformersMeanEmbedder",
    "build_embedder",
    "resolve_device",
]
