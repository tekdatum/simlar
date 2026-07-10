from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

# SearchResult and _Parameters are defined in simlar_engine and re-exported here
# so user code can import them from either package.
from simlar_engine._types import SearchResult, _Parameters

__all__ = [
    "SearchResult",
    "_Parameters",
    "Index",
    "TextIndex",
    "VectorIndex",
    "CompositeIndex",
    "FusionStrategy",
]

# ── Base ──────────────────────────────────────────────────────────────────────


class Index(ABC):
    """Minimal contract: every index can persist and report its type and size."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Index: ...

    @property
    @abstractmethod
    def index_type(self) -> str: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def is_trained(self) -> bool: ...


# ── Text index ────────────────────────────────────────────────────────────────


class TextIndex(Index):
    """Index over a text corpus."""

    # ── Public API ─────────────────────────────────────────────────────────────

    @abstractmethod
    def add(self, ids: list[str], texts: list[str]) -> None:
        """Append new documents. Raises ValueError on duplicate IDs; use update() to replace."""

    @abstractmethod
    def update(self, ids: list[str], texts: list[str]) -> None:
        """Replace texts for existing document IDs, then rebuild."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove documents by ID, rebuilding internal structures."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[SearchResult]: ...

    # ── Internal ───────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, corpus: list[str], parallel: bool = False, **kwargs) -> None:
        """Build the model from a raw corpus list. Called by add() and HelixIndex."""

    @abstractmethod
    def search_raw(
        self,
        queries: str | list[str],
        k: int,
        parallel: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores) shaped (n_queries, k), int64/float64.
        Used internally by HelixIndex._search_raw().
        """


# ── Vector index ──────────────────────────────────────────────────────────────


class VectorIndex(Index):
    """Index over dense or binary float embeddings."""

    # ── Public API ─────────────────────────────────────────────────────────────

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[SearchResult]: ...

    @abstractmethod
    def update(self, ids: list[str], vectors: np.ndarray) -> None:
        """Replace vectors for existing IDs in-place using frozen quantization params."""

    # ── Internal ───────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        embeddings: np.ndarray,
        parallel: bool = False,
        params: _Parameters | None = None,
        **kwargs,
    ) -> None:
        """Build from embeddings. Pass params to reuse frozen quantization across shards."""

    @abstractmethod
    def search_raw(
        self,
        vectors: np.ndarray,
        k: int,
        candidates: np.ndarray | None = None,
        parallel: bool = False,
        n_candidates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) shaped (n_queries, k), int64/float64.
        Used internally by HelixIndex._search_raw().
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove documents by ID, physically rebuilding internal structures."""


# ── Composite index ───────────────────────────────────────────────────────────


class CompositeIndex(Index):
    """Fuses N sub-indexes with a FusionStrategy."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        texts: list[str] | None = None,
        vectors: np.ndarray | None = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_text: str | None = None,
        query_vector: np.ndarray | None = None,
        k: int = 10,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def fit(
        self,
        corpus: list,
        vectors: np.ndarray,
        parallel: bool = False,
        **kwargs,
    ) -> None:
        """Internal: build all sub-indexes. Used by StreamingHybridIndex shards."""


# ── Fusion ────────────────────────────────────────────────────────────────────


@runtime_checkable
class FusionStrategy(Protocol):
    """Combines N result lists into a single ranked list."""

    def __call__(
        self,
        results: list[list[SearchResult]],
        k: int,
    ) -> list[SearchResult]: ...
