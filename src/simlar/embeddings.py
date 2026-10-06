"""Pluggable text embedders.

An index built with ``embedder=`` turns texts into vectors itself, so
``add``/``update``/``fit``/``search`` can be called with text alone::

    from sentence_transformers import SentenceTransformer
    from simlar import HelixIndex

    model = SentenceTransformer("all-MiniLM-L6-v2")
    idx = HelixIndex(embedder=model.encode)
    idx.add(["a", "b"], ["apple pie", "car engine"])
    idx.search("dessert", k=5)

Any object with ``embed_documents`` / ``embed_query`` (a LangChain
``Embeddings`` included) satisfies ``Embedder``; a bare callable
``fn(list[str]) -> array`` is wrapped in ``CallableEmbedder``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors. Return values may be arrays or nested lists."""

    def embed_documents(self, texts: list[str]) -> Any: ...

    def embed_query(self, text: str) -> Any: ...


class CallableEmbedder:
    """Wrap ``fn(list[str]) -> (n, dim) array-like`` as an ``Embedder``.

    Args:
        fn: Embeds a list of documents.
        query_fn: Embeds one query string; defaults to ``fn([text])[0]``.
        normalize: L2-normalize every vector.
        batch_size: Call ``fn`` on chunks of at most this many texts.
    """

    def __init__(
        self,
        fn: Callable[[list[str]], Any],
        query_fn: Callable[[str], Any] | None = None,
        *,
        normalize: bool = False,
        batch_size: int | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._fn = fn
        self._query_fn = query_fn
        self.normalize = normalize
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        if self.batch_size is None or len(texts) <= self.batch_size:
            out = _as_matrix(self._fn(texts), len(texts))
        else:
            step = self.batch_size
            out = np.concatenate(
                [
                    _as_matrix(self._fn(texts[i : i + step]), len(texts[i : i + step]))
                    for i in range(0, len(texts), step)
                ]
            )
        return _l2_normalize(out) if self.normalize else out

    def embed_query(self, text: str) -> np.ndarray:
        if self._query_fn is None:
            vec = _as_matrix(self._fn([text]), 1)[0]
        else:
            vec = _as_matrix(self._query_fn(text), 1)[0]
        return _l2_normalize(vec[None, :])[0] if self.normalize else vec

    def __repr__(self) -> str:
        name = getattr(self._fn, "__qualname__", type(self._fn).__name__)
        return f"CallableEmbedder({name}, normalize={self.normalize}, batch_size={self.batch_size})"


def as_embedder(obj: Any) -> Embedder | None:
    """Return *obj* as an ``Embedder``: pass through ``None`` and embedders,
    wrap a bare callable, reject anything else."""
    if obj is None or isinstance(obj, Embedder):
        return obj
    if callable(obj):
        return CallableEmbedder(obj)
    raise TypeError(
        f"embedder must have embed_documents/embed_query or be callable, got {type(obj).__name__}"
    )


# ── Index-side helpers ────────────────────────────────────────────────────────


def _as_matrix(values: Any, n: int) -> np.ndarray:
    """Coerce embedder output to a C-contiguous (n, dim) float32 array."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if arr.ndim == 1 and n == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"embedder returned an array of shape {arr.shape}; expected (n, dim)")
    if arr.shape[0] != n:
        raise ValueError(f"embedder returned {arr.shape[0]} vectors for {n} texts")
    return arr


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _require(embedder: Embedder | None, what: str) -> Embedder:
    if embedder is None:
        raise ValueError(f"{what} is required: this index has no embedder to compute it from text")
    return embedder


def _embed_documents(embedder: Embedder | None, texts: Sequence[str]) -> np.ndarray:
    texts = list(texts)
    return _as_matrix(_require(embedder, "vectors").embed_documents(texts), len(texts))


def _embed_queries(embedder: Embedder | None, query_text: str | Sequence[str]) -> np.ndarray:
    """A ``str`` becomes a 1D vector; a list becomes an (n, dim) batch."""
    e = _require(embedder, "a query vector")
    if isinstance(query_text, str):
        return _as_matrix(e.embed_query(query_text), 1)[0]
    queries = list(query_text)
    return np.stack([_as_matrix(e.embed_query(q), 1)[0] for q in queries])


def _resolve_vectors(
    embedder: Embedder | None, vectors: np.ndarray | None, texts: Sequence[str] | None
) -> np.ndarray | None:
    """Return *vectors*, or embed *texts* when vectors are missing and an embedder is set."""
    if vectors is not None or texts is None or embedder is None:
        return vectors
    return _embed_documents(embedder, texts)
