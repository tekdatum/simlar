from __future__ import annotations

import numpy as np

from simlar.contracts import SearchResult, VectorIndex, _Parameters

class SimlarEngine(VectorIndex):
    """
    Example::

        idx = SimlarEngine()
        idx.fit(embeddings=vecs)
        idx.add(ids=["a", "b"], vectors=vecs)
        results = idx.search(query=q_vec, k=10)
    """

    def __init__(self, n_candidates: int | None = None) -> None: ...
    def fit(
        self,
        embeddings: np.ndarray,
        parallel: bool = True,
        params: _Parameters | None = None,
        **kwargs: object,
    ) -> None: ...
    def add(self, ids: list[str], vectors: np.ndarray, parallel: bool = True) -> None: ...
    def update(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        parallel: bool = True,
        batch_size: int | None = None,
    ) -> list[SearchResult]: ...
    def search_raw(
        self,
        vectors: np.ndarray,
        k: int,
        candidates: np.ndarray | None = None,
        parallel: bool = True,
        n_candidates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def save(self, directory: str, base_dir: str | None = None) -> None: ...
    @classmethod
    def load(cls, directory: str, base_dir: str | None = None) -> SimlarEngine: ...
    @property
    def size(self) -> int: ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def index_type(self) -> str: ...
    @property
    def coreindex(self) -> object: ...
    @property
    def boundaries(self) -> np.ndarray | None: ...
    @property
    def fit_values(self) -> np.ndarray | None: ...
    @property
    def _params(self) -> _Parameters | None: ...
    @property
    def _matrix(self) -> np.ndarray | None: ...
    @property
    def ids(self) -> list[str]: ...
