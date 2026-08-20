from __future__ import annotations

import numpy as np
from simlar_engine.indexes._simlar_impl import _SimlarCore

from simlar.contracts import SearchResult, VectorIndex, _Parameters
from simlar.indexes.registry import register


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
    """

    def __init__(self, n_candidates: int | None = None) -> None:
        self._core = _SimlarCore(n_candidates)

    # ── Public contract ───────────────────────────────────────────────────────

    def fit(
        self,
        embeddings: np.ndarray,
        parallel: bool = True,
        params: _Parameters | None = None,
        **kwargs,
    ) -> None:
        self._core.fit(embeddings, parallel, params)

    def add(self, ids: list[str], vectors: np.ndarray, parallel: bool = True) -> None:
        self._core.add(ids, vectors, parallel)

    def update(self, ids: list[str], vectors: np.ndarray) -> None:
        self._core.update(ids, vectors)

    def delete(self, ids: list[str]) -> None:
        self._core.delete(ids)

    def search(self, query: np.ndarray, k: int = 10, parallel: bool = True) -> list[SearchResult]:
        return self._core.search(query, k, parallel)

    def search_raw(
        self,
        vectors: np.ndarray,
        k: int,
        candidates: np.ndarray | None = None,
        parallel: bool = True,
        n_candidates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._core.search_raw(vectors, k, candidates, parallel, n_candidates)

    def save(self, directory: str) -> None:
        self._core.save(directory)

    @classmethod
    def load(cls, directory: str) -> SimlarEngine:
        obj = cls.__new__(cls)
        obj._core = _SimlarCore.load(directory)
        return obj

    # ── Properties ────────────────────────────────────────────────────────────

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
