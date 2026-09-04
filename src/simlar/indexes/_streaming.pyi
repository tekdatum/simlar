from __future__ import annotations

import numpy as np

from simlar.contracts import TextIndex, VectorIndex

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
        text_k: int | None = None,
        vector_k: int | None = None,
        top_k: int = 100,
        alpha_text: float = 0.10,
        alpha_vector: float = 1.0,
        rrf_k: int = 2,
        n_candidates: int | None = None,
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
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def save(self, directory: str, base_dir: str | None = None) -> None: ...
    @classmethod
    def load(cls, directory: str, base_dir: str | None = None) -> StreamingHybridIndex: ...
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
