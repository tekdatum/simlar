from __future__ import annotations

import threading

import numpy as np

from simlar._engine import engine_core, require_core
from simlar.contracts import SearchResult, VectorIndex, _Parameters
from simlar.embeddings import Embedder, _embed_documents, _embed_queries, as_embedder
from simlar.indexes.registry import register

_SimlarCore = engine_core("simlar", "simlar_engine.indexes._simlar_impl", "_SimlarCore")


class _RWLock:
    """Readers run concurrently; a writer waits for all readers to drain and
    excludes both other writers and new readers for its duration.

    Exists because `_SimlarCore.delete()`/`add()`/`update()` mutate shared
    state (`_ids`, `_coreindex`, the byte matrix) in several separate,
    unlocked steps — a `search()`/`save()` landing in the middle of one of
    those reads a torn mix of pre- and post-mutation state and segfaults
    (see docs/issue-04-delete-search-race.md). This lock is the mutual
    exclusion the engine itself has none of.
    """

    def __init__(self) -> None:
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self) -> None:
        with self._read_ready:
            self._readers += 1

    def release_read(self) -> None:
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self) -> None:
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self) -> None:
        self._read_ready.release()


@register("simlar")
class SimlarEngine(VectorIndex):
    """
    Example:
        >>> import numpy as np
        >>> idx = SimlarEngine()
        >>> embeddings = np.random.rand(1000, 128).astype(np.float32)
        >>> idx.fit(embeddings)
        >>> query = np.random.rand(128).astype(np.float32)
        >>> results = idx.search(query, k=10)

    With an ``embedder`` the index embeds texts itself::

        >>> idx = SimlarEngine(embedder=model.encode)
        >>> idx.add(["a", "b"], texts=["apple pie", "car engine"])
        >>> idx.search(query_text="dessert", k=5)
    """

    def __init__(self, n_candidates: int | None = None, *, embedder=None) -> None:
        self._core = require_core("simlar", _SimlarCore)(n_candidates)
        self._rwlock = _RWLock()
        self._embedder = as_embedder(embedder)

    # ── Public contract ───────────────────────────────────────────────────────

    def fit(
        self,
        embeddings: np.ndarray | None = None,
        parallel: bool = True,
        params: _Parameters | None = None,
        *,
        texts: list[str] | None = None,
        **kwargs,
    ) -> None:
        # Embed outside the lock: a slow embedder must not stall readers.
        embeddings = self._vectors(embeddings, texts)
        self._rwlock.acquire_write()
        try:
            self._core.fit(embeddings, parallel, params)
        finally:
            self._rwlock.release_write()

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray | None = None,
        parallel: bool = True,
        *,
        texts: list[str] | None = None,
    ) -> None:
        vectors = self._vectors(vectors, texts)
        self._rwlock.acquire_write()
        try:
            self._core.add(ids, vectors, parallel)
        finally:
            self._rwlock.release_write()

    def update(
        self,
        ids: list[str],
        vectors: np.ndarray | None = None,
        *,
        texts: list[str] | None = None,
    ) -> None:
        vectors = self._vectors(vectors, texts)
        self._rwlock.acquire_write()
        try:
            self._core.update(ids, vectors)
        finally:
            self._rwlock.release_write()

    def delete(self, ids: list[str]) -> None:
        self._rwlock.acquire_write()
        try:
            self._core.delete(ids)
        finally:
            self._rwlock.release_write()

    def search(
        self,
        query: np.ndarray | None = None,
        k: int = 10,
        parallel: bool = True,
        batch_size: int | None = None,
        candidates: np.ndarray | None = None,
        *,
        query_text: str | list[str] | None = None,
    ) -> list[SearchResult]:
        if query is not None and query_text is not None:
            raise ValueError("Pass either query or query_text, not both")
        if query is None:
            if query_text is None:
                raise ValueError("search() needs query or query_text")
            query = _embed_queries(self.embedder, query_text)
        self._rwlock.acquire_read()
        try:
            return self._core.search(query, k, parallel, batch_size, candidates)
        finally:
            self._rwlock.release_read()

    def search_raw(
        self,
        vectors: np.ndarray,
        k: int,
        candidates: np.ndarray | None = None,
        parallel: bool = True,
        n_candidates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._rwlock.acquire_read()
        try:
            return self._core.search_raw(vectors, k, candidates, parallel, n_candidates)
        finally:
            self._rwlock.release_read()

    def save(self, directory: str, base_dir: str | None = None) -> None:
        self._rwlock.acquire_read()
        try:
            self._core.save(directory, base_dir)
        finally:
            self._rwlock.release_read()

    @classmethod
    def load(cls, directory: str, base_dir: str | None = None, *, embedder=None) -> SimlarEngine:
        obj = cls.__new__(cls)
        obj._core = require_core("simlar", _SimlarCore).load(directory, base_dir)
        obj._rwlock = _RWLock()
        obj._embedder = as_embedder(embedder)
        return obj

    def _vectors(self, vectors: np.ndarray | None, texts: list[str] | None) -> np.ndarray:
        if vectors is not None and texts is not None:
            raise ValueError("Pass either vectors or texts, not both")
        if vectors is not None:
            return vectors
        if texts is None:
            raise ValueError("vectors or texts is required")
        return _embed_documents(self.embedder, texts)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def embedder(self) -> Embedder | None:
        return getattr(self, "_embedder", None)

    @embedder.setter
    def embedder(self, value) -> None:
        self._embedder = as_embedder(value)

    @property
    def size(self) -> int:
        return self._core.size

    @property
    def is_trained(self) -> bool:
        return self._core.is_trained

    @property
    def index_type(self) -> str:
        return "simlar"

    @property
    def coreindex(self):
        return self._core.coreindex

    @property
    def _params(self) -> _Parameters | None:
        return self._core._params

    @property
    def boundaries(self) -> np.ndarray | None:
        p = self._core._params
        return p.boundaries if p is not None else None

    @property
    def fit_values(self) -> np.ndarray | None:
        p = self._core._params
        return p.fit_values if p is not None else None

    @property
    def _matrix(self) -> np.ndarray | None:
        return self._core._matrix

    @property
    def ids(self) -> list[str]:
        return self._core.ids
