from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import bm25x
except ImportError:  # pragma: no cover - exercised via the import-guard test
    bm25x = None  # type: ignore[assignment]

from simlar.contracts import SearchResult, TextIndex
from simlar.indexes.registry import register
from simlar.persistence import read_config, write_config

_BM25X_MISSING = (
    "bm25x is not installed. As of this writing, bm25x only ships compiled wheels for "
    "Python 3.12 (no wheels for other versions, and no source distribution to build from), "
    "so it's an optional, version-gated extra rather than a hard dependency of simlar. "
    "Install it with `pip install simlar[bm25x]` on Python 3.12."
)


def _require_bm25x() -> None:
    if bm25x is None:
        raise ImportError(_BM25X_MISSING)


@register("bm25x")
class BM25xIndex(TextIndex):
    """A `TextIndex` backed by the open-source `bm25x` library, independent of `simlar_engine`.

    Example:
        >>> idx = BM25xIndex()
        >>> idx.add(["a", "b"], ["hello world", "foo bar"])
        >>> results = idx.search("hello", k=5)
        >>> results[0].id
        'a'
    """

    def __init__(
        self,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ) -> None:
        _require_bm25x()
        self._method = method
        self._k1 = k1
        self._b = b
        self._delta = delta
        self._core = bm25x.BM25(method=method, k1=k1, b=b, delta=delta)
        self._ids: list[str] = []

    # ── Public contract ───────────────────────────────────────────────────────

    def fit(self, corpus: list[str], parallel: bool = False, **kwargs: object) -> None:
        if not corpus:
            raise ValueError("corpus cannot be empty")
        self._core = bm25x.BM25(method=self._method, k1=self._k1, b=self._b, delta=self._delta)
        self._core.add(corpus)
        self._ids = [str(i) for i in range(len(corpus))]

    def add(self, ids: list[str], texts: list[str]) -> None:
        existing = set(self._ids)
        seen_in_batch: set[str] = set()
        for doc_id in ids:
            if doc_id in existing:
                raise ValueError(f"duplicate id {doc_id!r}: already present, use update() instead")
            if doc_id in seen_in_batch:
                raise ValueError(f"duplicate id {doc_id!r} within this add() call")
            seen_in_batch.add(doc_id)
        self._core.add(texts)
        self._ids.extend(ids)

    def update(self, ids: list[str], texts: list[str]) -> None:
        position_by_id = self._position_by_id()
        for doc_id in ids:
            if doc_id not in position_by_id:
                raise ValueError(
                    f"unknown id {doc_id!r}: cannot update a document that was never added"
                )
        for doc_id, text in zip(ids, texts, strict=True):
            self._core.update(position_by_id[doc_id], text)

    def delete(self, ids: list[str]) -> None:
        position_by_id = self._position_by_id()
        positions = []
        for doc_id in ids:
            if doc_id not in position_by_id:
                raise ValueError(
                    f"unknown id {doc_id!r}: cannot delete a document that was never added"
                )
            positions.append(position_by_id[doc_id])
        self._core.delete(positions)
        deleted = set(positions)
        self._ids = [doc_id for i, doc_id in enumerate(self._ids) if i not in deleted]

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if not self.is_trained:
            raise ValueError("cannot search an empty/untrained index - call fit() or add() first")
        raw = self._core.search(query, k)
        return [
            SearchResult(rank=rank, id=self._ids[position], score=score)
            for rank, (position, score) in enumerate(raw)
        ]

    def search_raw(
        self,
        queries: str | list[str],
        k: int,
        parallel: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError("cannot search an empty/untrained index - call fit() or add() first")
        # bm25x has no `parallel` toggle of its own - batching multiple queries in one call is
        # already its parallel path (rayon-parallelized internally), so there's nothing to forward.
        if isinstance(queries, str):
            raw = self._core.search(queries, k)
            return self._pad_row(raw, k)
        rows = self._core.search(queries, k)
        ids = np.empty((len(rows), k), dtype=np.int64)
        scores = np.empty((len(rows), k), dtype=np.float64)
        for i, row in enumerate(rows):
            row_ids, row_scores = self._pad_row(row, k)
            ids[i] = row_ids
            scores[i] = row_scores
        return ids, scores

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        write_config(
            d / "config.json",
            {
                "index_type": "bm25x",
                "method": self._method,
                "k1": self._k1,
                "b": self._b,
                "delta": self._delta,
            },
        )
        (d / "ids.json").write_text(json.dumps(self._ids))
        native_dir = d / "bm25x_native"
        native_dir.mkdir(parents=True, exist_ok=True)
        self._core.save(str(native_dir))

    @classmethod
    def load(cls, directory: str) -> BM25xIndex:
        _require_bm25x()
        d = Path(directory)
        meta = read_config(d / "config.json")
        obj = cls.__new__(cls)
        obj._method = meta["method"]
        obj._k1 = meta["k1"]
        obj._b = meta["b"]
        obj._delta = meta["delta"]
        obj._ids = json.loads((d / "ids.json").read_text())
        obj._core = bm25x.BM25.load(str(d / "bm25x_native"))
        return obj

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def is_trained(self) -> bool:
        return len(self._ids) > 0

    @property
    def index_type(self) -> str:
        return "bm25x"

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _position_by_id(self) -> dict[str, int]:
        return {doc_id: position for position, doc_id in enumerate(self._ids)}

    @staticmethod
    def _pad_row(row: list[tuple[int, float]], k: int) -> tuple[np.ndarray, np.ndarray]:
        """bm25x.search() returns fewer than k tuples when fewer than k documents match at
        all - pad with (-1, -inf) so callers get a fixed-width row, matching the (n_queries, k)
        shape TextIndex.search_raw's contract documents."""
        ids = np.full(k, -1, dtype=np.int64)
        scores = np.full(k, -np.inf, dtype=np.float64)
        for i, (position, score) in enumerate(row):
            ids[i] = position
            scores[i] = score
        return ids, scores
