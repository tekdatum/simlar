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
from simlar.embeddings import Embedder, _embed_queries, _resolve_vectors, as_embedder
from simlar.indexes.registry import register


@register("helix")
class HelixIndex(CompositeIndex):
    """Fuses N indexes with a FusionStrategy.

    Example::

        HelixIndex(indexes=[RelevanceIndex(), SimilarityIndex()], fusion=ReciprocalRankFusion())

    With an ``embedder``, texts given without vectors are embedded for the
    vector side, and a ``query_text`` without ``query_vector`` is embedded
    too, so text-only calls search both signals::

        idx = HelixIndex(embedder=model.encode)
        idx.add(["a", "b"], ["apple pie", "car engine"])
        idx.search("dessert", k=5)

    Without one, text-only searches use the text index alone.

    ``text_k`` / ``vector_k`` left unset are tuned to the corpus size and
    ``top_k`` from a model shipped with the engine (for relevance and lookup
    text indexes). ``speed_preference`` trades latency for recall:
    "fastest", "fast", "balanced" (default), "accurate", "most accurate".
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
        speed_preference: str = "balanced",
        embedder=None,
    ) -> None:
        self._embedder = as_embedder(embedder)
        self._core = _HelixCore(
            text_index=text_index,
            vector_index=vector_index,
            fusion=fusion,
            text_k=text_k,
            vector_k=vector_k,
            top_k=top_k,
            alpha_text=alpha_text,
            alpha_vector=alpha_vector,
            speed_preference=speed_preference,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        ids: list[str],
        texts: list[str] | None = None,
        vectors: np.ndarray | None = None,
        parallel: bool = True,
    ) -> None:
        vectors = _resolve_vectors(self.embedder, vectors, texts)
        self._core.add(ids, texts, vectors, parallel)

    def update(
        self,
        ids: list[str],
        texts: list[str] | None = None,
        vectors: np.ndarray | None = None,
    ) -> None:
        """Replace the text and/or vector of existing documents; positions are unchanged.

        With an embedder, new texts given without vectors are re-embedded so
        both sides stay in step.
        """
        vectors = _resolve_vectors(self.embedder, vectors, texts)
        self._core.update(ids, texts, vectors)

    def delete(self, ids: list[str]) -> None:
        """Delete documents from both sub-indexes, compacting positions."""
        self._core.delete(ids)

    # A list of texts or a 2D query_vector is a batch and returns one result
    # list per query, which the single-query CompositeIndex contract can't express.
    def search(  # type: ignore[override]
        self,
        query_text: str | list[str] | None = None,
        query_vector: np.ndarray | None = None,
        k: int | None = None,
        parallel: bool = True,
        batch_size: int | None = None,
        candidates: np.ndarray | None = None,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        if query_vector is None and query_text is not None and self.embedder is not None:
            query_vector = _embed_queries(self.embedder, query_text)
        return self._core.search(query_text, query_vector, k, parallel, batch_size, candidates)

    def fit(
        self,
        corpus: list,
        vectors: np.ndarray | None = None,
        parallel: bool = True,
        **kwargs,
    ) -> None:
        params = kwargs.pop("params", None)
        vectors = _resolve_vectors(self.embedder, vectors, corpus)
        if vectors is None:
            raise ValueError(
                "vectors is required: this index has no embedder to compute it from text"
            )
        self._core.fit(corpus, vectors, parallel, params)

    def save(self, directory: str, base_dir: str | None = None) -> None:
        self._core.save(directory, base_dir)

    @classmethod
    def load(cls, directory: str, base_dir: str | None = None, *, embedder=None) -> HelixIndex:
        obj = cls.__new__(cls)
        obj._core = _HelixCore.load(directory, base_dir)
        obj._embedder = as_embedder(embedder)
        return obj

    # ── Metadata ──────────────────────────────────────────────────────────────

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

    @property
    def speed_preference(self) -> str:
        return self._core.speed_preference

    @property
    def expected_recall(self) -> float | None:
        """Recall the tuned model expects of the auto-tuned text_k/vector_k;
        None when they weren't tuned or the corpus is too small to vouch for."""
        return self._core.expected_recall

    @property
    def ids(self) -> list[str]:
        return self.text_index.ids

    def __repr__(self) -> str:
        return (
            f"HelixIndex(text={self._core.text_index.index_type!r}, "
            f"vector={self._core.vector_index.index_type!r}, "
            f"text_k={self._core.text_k}, vector_k={self._core.vector_k})"
        )
