"""Indexing stage: exact cosine FAISS index over chunk vectors."""

from depression_rag.indexing.faiss_index import FaissFlatIndex, build_flat_index

__all__ = ["FaissFlatIndex", "build_flat_index"]
