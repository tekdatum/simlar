from __future__ import annotations

import numpy as np

from simlar.contracts import TextIndex, VectorIndex, _Parameters

class StreamingHybridIndex:
    """
    Example::

        idx = StreamingHybridIndex()
        idx.add_batch(corpus=texts[:1000], vectors=vecs[:1000])
        idx.add_batch(corpus=texts[1000:], vectors=vecs[1000:])
        ids, distances = idx.search(query_text="query", query_vector=q_vec, k=10)
        idx.save("streaming.idx")
        idx = StreamingHybridIndex.load("streaming.idx")
    """

    def __init__(
        self,
        text_index_cls: type[TextIndex] | None = None,
        vector_index_cls: type[VectorIndex] | None = None,
        text_k: int = 5000,
        vector_k: int = 1000,
        top_k: int = 100,
        alpha_text: float = 0.10,
        alpha_vector: float = 1.0,
        rrf_k: int = 2,
        n_candidates: int = 5000,
    ) -> None: ...
    def add_batch(self, corpus: list[str], vectors: np.ndarray, parallel: bool = False) -> None: ...
    async def add_batch_async(
        self, corpus: list[str], vectors: np.ndarray, parallel: bool = False
    ) -> None: ...
    def search(
        self,
        query_text: str | list[str],
        query_vector: np.ndarray,
        k: int | None = None,
        parallel: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def fit(self, corpus: list, vectors: np.ndarray, **kwargs: object) -> None: ...
    def save(self, directory: str) -> None: ...
    @classmethod
    def load(cls, directory: str) -> StreamingHybridIndex: ...
    @property
    def size(self) -> int: ...
    @property
    def n_shards(self) -> int: ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def index_type(self) -> str: ...
    @property
    def boundaries(self) -> np.ndarray | None: ...
    @property
    def fit_values(self) -> np.ndarray | None: ...
    @property
    def _params(self) -> _Parameters | None: ...
