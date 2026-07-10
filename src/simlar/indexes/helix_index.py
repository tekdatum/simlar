from __future__ import annotations

import numpy as np
from simlar_engine.indexes._helix_impl import _HelixCore

from simlar.contracts import (
    CompositeIndex,
    FusionStrategy,
    SearchResult,
    TextIndex,
    VectorIndex,
    _Parameters,
)
from simlar.indexes.registry import register


@register("helix")
class HelixIndex(CompositeIndex):
    """Fuses N indexes with a FusionStrategy.

    Example::

        HelixIndex(indexes=[RelevanceIndex(), SimilarityIndex()], fusion=ReciprocalRankFusion())
    """

    def __init__(
        self,
        *,
        text_index: TextIndex | None = None,
        vector_index: VectorIndex | None = None,
        fusion: FusionStrategy | None = None,
        text_k: int | None = None,
        vector_k: int | None = None,
        top_k: int = 100,
        alpha_text: float = 0.10,
        alpha_vector: float = 1.0,
    ) -> None:
        self._core = _HelixCore(
            text_index=text_index,
            vector_index=vector_index,
            fusion=fusion,
            text_k=text_k,
            vector_k=vector_k,
            top_k=top_k,
            alpha_text=alpha_text,
            alpha_vector=alpha_vector,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        ids: list[str],
        texts: list[str] | None = None,
        vectors: np.ndarray | None = None,
    ) -> None:
        self._core.add(ids, texts, vectors)

    def search(
        self,
        query_text: str | list[str] | None = None,
        query_vector: np.ndarray | None = None,
        k: int | None = None,
        parallel: bool = False,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        return self._core.search(query_text, query_vector, k, parallel)

    def fit(
        self,
        corpus: list,
        vectors: np.ndarray,
        parallel: bool = False,
        **kwargs,
    ) -> None:
        params = kwargs.pop("params", None)
        self._core.fit(corpus, vectors, parallel, params)

    def save(self, directory: str) -> None:
        self._core.save(directory)

    @classmethod
    def load(cls, directory: str) -> HelixIndex:
        obj = cls.__new__(cls)
        obj._core = _HelixCore.load(directory)
        return obj

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._core.size

    @property
    def is_trained(self) -> bool:
        return self._core.is_trained

    @property
    def index_type(self) -> str:
        return "helix"

    @property
    def boundaries(self) -> np.ndarray | None:
        return self._core.boundaries

    @property
    def fit_values(self) -> np.ndarray | None:
        return self._core.fit_values

    @property
    def _params(self) -> _Parameters | None:
        return self._core._params

    @property
    def text_index(self) -> TextIndex:
        return self._core.text_index

    @property
    def vector_index(self) -> VectorIndex:
        return self._core.vector_index

    def __repr__(self) -> str:
        return (
            f"HelixIndex(text={self._core.text_index.index_type!r}, "
            f"vector={self._core.vector_index.index_type!r}, "
            f"text_k={self._core._text_k}, vector_k={self._core._vector_k})"
        )
