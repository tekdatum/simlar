from __future__ import annotations

import numpy as np

from simlar.contracts import (
    CompositeIndex,
    FusionStrategy,
    SearchResult,
    TextIndex,
    VectorIndex,
    _Parameters,
)

class HelixIndex(CompositeIndex):
    """
    Example::

        from simlar import HelixIndex, RelevanceIndex, SimlarEngine
        from simlar.fusion import ReciprocalRankFusion

        idx = HelixIndex(
            text_index=RelevanceIndex(),
            vector_index=SimlarEngine(),
            fusion=ReciprocalRankFusion(),
        )
        idx.fit(corpus=texts, vectors=embeddings)
        idx.add(ids=ids, texts=texts, vectors=embeddings)
        results = idx.search(query_text="query", query_vector=q_vec, k=10)
    """

    def __init__(
        self,
        text_index: TextIndex | None = None,
        vector_index: VectorIndex | None = None,
        fusion: FusionStrategy | None = None,
        text_k: int = 5000,
        vector_k: int = 1000,
        top_k: int = 100,
        alpha_text: float = 0.10,
        alpha_vector: float = 1.0,
        rrf_k: int = 2,
    ) -> None: ...
    def fit(
        self,
        corpus: list[str],
        vectors: np.ndarray,
        parallel: bool = False,
        params: _Parameters | None = None,
        **kwargs: object,
    ) -> None: ...
    def add(
        self,
        ids: list[str],
        texts: list[str] | None = None,
        vectors: np.ndarray | None = None,
        parallel: bool = False,
    ) -> None: ...
    def search(
        self,
        query_text: str | list[str] | None = None,
        query_vector: np.ndarray | None = None,
        k: int | None = None,
        parallel: bool = False,
    ) -> list[SearchResult]: ...
    def save(self, directory: str) -> None: ...
    @classmethod
    def load(cls, directory: str) -> HelixIndex: ...
    @property
    def size(self) -> int: ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def index_type(self) -> str: ...
    @property
    def text_index(self) -> TextIndex: ...
    @property
    def vector_index(self) -> VectorIndex: ...
    @property
    def boundaries(self) -> np.ndarray | None: ...
    @property
    def fit_values(self) -> np.ndarray | None: ...
    @property
    def quantization_params(self) -> _Parameters | None: ...
    @property
    def ids(self) -> list[str]: ...
