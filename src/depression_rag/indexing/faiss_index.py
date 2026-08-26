"""FAISS exact index.

`IndexFlatIP` = brute-force inner product. Vectors are already L2-normalized by
the embedder, so inner product == cosine. No approximate index (HNSW/IVF) at
this corpus scale - exact search removes a source of variance from the
comparison. faiss is imported lazily so the package imports without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class FaissFlatIndex:
    """Exact inner-product index over chunk vectors (FAISS ``IndexFlatIP``).

    Multi-vector indexes repeat a ``chunk_id`` across several window vectors;
    :meth:`search` collapses duplicates so callers always see one entry per chunk.
    """

    def __init__(self, dim: int, metric: str = "ip") -> None:
        import faiss

        if metric != "ip":
            raise ValueError(f"only metric='ip' (cosine on normalized vectors) is supported, got {metric!r}")
        self._faiss = faiss
        self._dim = dim
        self._metric = metric
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._n_unique = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"expected (n, {self._dim}) vectors, got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("number of ids must match number of vectors")
        self._index.add(vectors)
        self._ids.extend(ids)
        self._n_unique = len(set(self._ids))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, list[list[str]]]:
        """Top-k UNIQUE ids per query, best first.

        Multi-vector indexes hold several vectors per chunk (same id repeated);
        duplicates are collapsed keeping the best-scoring window, so callers
        always see one entry per chunk. Single-vector indexes take the plain
        path and behave exactly as before.
        """
        queries = np.ascontiguousarray(queries, dtype="float32")
        k = min(k, max(self.size, 1))
        if self._n_unique == len(self._ids):  # no duplicate ids: plain search
            scores, idx = self._index.search(queries, k)
            ids = [[self._ids[i] for i in row if i != -1] for row in idx]
            return scores, ids
        # over-fetch so the top-k unique ids are guaranteed present, then dedup
        k_out = min(k, self._n_unique)
        k_raw = min(self.size, k_out + (len(self._ids) - self._n_unique))
        raw_scores, raw_idx = self._index.search(queries, k_raw)
        out_scores = np.full((queries.shape[0], k_out), -np.inf, dtype="float32")
        out_ids: list[list[str]] = []
        for r, (srow, irow) in enumerate(zip(raw_scores, raw_idx)):
            seen: set[str] = set()
            row_ids: list[str] = []
            for s, i in zip(srow, irow):
                if i == -1 or self._ids[i] in seen:
                    continue
                seen.add(self._ids[i])
                out_scores[r, len(row_ids)] = s
                row_ids.append(self._ids[i])
                if len(row_ids) == k_out:
                    break
            out_ids.append(row_ids)
        return out_scores, out_ids

    def save(self, directory: Path | str) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "index.faiss"
        ids_path = directory / "ids.json"
        self._faiss.write_index(self._index, str(index_path))
        ids_path.write_text(
            json.dumps({"dim": self._dim, "metric": self._metric, "ids": self._ids},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        return [index_path, ids_path]

    @classmethod
    def load(cls, directory: Path | str) -> "FaissFlatIndex":
        import faiss

        directory = Path(directory)
        meta = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        obj = cls(int(meta["dim"]), meta.get("metric", "ip"))
        obj._index = faiss.read_index(str(directory / "index.faiss"))
        obj._ids = list(meta["ids"])
        obj._n_unique = len(set(obj._ids))
        return obj


def build_flat_index(vectors: np.ndarray, ids: Sequence[str], metric: str = "ip") -> FaissFlatIndex:
    """Convenience: create a FaissFlatIndex sized to the vectors and add them."""
    index = FaissFlatIndex(int(vectors.shape[1]), metric=metric)
    index.add(vectors, ids)
    return index
