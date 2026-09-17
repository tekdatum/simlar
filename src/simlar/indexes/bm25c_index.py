from __future__ import annotations

import numpy as np

try:
    import bm25c
except ImportError:  # pragma: no cover - exercised via the import-guard test
    bm25c = None  # type: ignore[assignment]

from simlar.contracts import SearchResult, TextIndex
from simlar.indexes.registry import register

_BM25C_MISSING = (
    "bm25c is not installed. It's an external, optional dependency (a from-scratch C "
    "implementation of BM25) -- see its own repo for build/install instructions. Hardcoded "
    "here for stress-testing purposes, not (yet) a packaged simlar extra."
)


def _require_bm25c() -> None:
    if bm25c is None:
        raise ImportError(_BM25C_MISSING)


@register("bm25c")
class BM25CIndex(TextIndex):
    """A `TextIndex` backed by the `bm25c` library, independent of `simlar_engine`/`bm25s`.

    Thin wrapper: `bm25c.BM25CRelevanceCore` already implements this
    exact contract shape (it was designed to mirror `_RelevanceCore`'s interface), so this
    class mostly just delegates -- the one real piece of work is converting bm25c's own
    SearchResult (a plain, separate class in bm25c.relevance_core) into
    simlar.contracts.SearchResult (== simlar_engine._types.SearchResult, the shared type
    the rest of this ecosystem expects, e.g. for isinstance checks).

    Example:
        >>> idx = BM25CIndex()
        >>> idx.add(["a", "b"], ["hello world", "foo bar"])
        >>> results = idx.search("hello", k=5)
        >>> results[0].id
        'a'
    """

    def __init__(
        self,
        method: str = "robertson",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords_lang: str = "english",
        stemmer_lang: str = "english",
    ) -> None:
        _require_bm25c()
        self._core = bm25c.BM25CRelevanceCore(
            method=method, k1=k1, b=b, stopwords_lang=stopwords_lang, stemmer_lang=stemmer_lang,
        )

    # ── Public contract ───────────────────────────────────────────────────────

    def fit(self, corpus: list[str], parallel: bool = False, **kwargs: object) -> None:
        self._core.fit(corpus)

    def add(self, ids: list[str], texts: list[str], parallel: bool = False) -> None:
        # bm25c's own add() has no parallel toggle (BM25 indexing is single-threaded in
        # every text backend, same as _RelevanceCore -- see RelevanceIndex.add()'s
        # matching fix) -- accepted here only for TextIndex interface conformance.
        self._core.add(ids, texts)

    def update(self, ids: list[str], texts: list[str]) -> None:
        self._core.update(ids, texts)

    def delete(self, ids: list[str]) -> None:
        self._core.delete(ids)

    def search(
        self,
        query: str,
        k: int = 10,
        parallel: bool = False,
        candidates: np.ndarray | None = None,
    ) -> list[SearchResult]:
        return [
            SearchResult(rank=r.rank, id=r.id, score=r.score, text=r.text)
            for r in self._core.search(query, k, candidates=candidates)
        ]

    def search_raw(
        self,
        queries: str | list[str],
        k: int,
        parallel: bool = False,
        candidates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`candidates`, if given, is a single flat array of positions applied
        to every query in this call (matching every other TextIndex backend's
        1D-only candidates contract). Filtering is purely a top-k
        selection-time concern for bm25c -- it costs no more accumulation
        work than an unrestricted search, only the final ranking differs."""
        return self._core.search_raw(queries, k, parallel, candidates=candidates)

    def save(self, directory: str) -> None:
        self._core.save(directory)

    @classmethod
    def load(cls, directory: str) -> BM25CIndex:
        _require_bm25c()
        obj = cls.__new__(cls)
        obj._core = bm25c.BM25CRelevanceCore.load(directory)
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
        return "bm25c"

    @property
    def ids(self) -> list[str]:
        return self._core.ids
