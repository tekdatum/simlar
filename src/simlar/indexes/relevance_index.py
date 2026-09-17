from __future__ import annotations

import numpy as np
from simlar_engine.indexes._relevance_impl import _RelevanceCore

from simlar.contracts import SearchResult, TextIndex
from simlar.indexes.registry import register


@register("relevance")
class RelevanceIndex(TextIndex):
    """
    Example:
        >>> idx = RelevanceIndex()
        >>> idx.fit(["hello world", "foo bar baz"])
        >>> results = idx.search("hello", k=5)
        >>> results[0].id
        '0'
    """

    def __init__(
        self,
        method: str = "robertson",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords_lang: str = "english",
        stemmer_lang: str = "english",
    ) -> None:
        self._core = _RelevanceCore(method, k1, b, stopwords_lang, stemmer_lang)

    # ── Public contract ───────────────────────────────────────────────────────

    def fit(self, corpus: list[str], parallel: bool = True, **kwargs: object) -> None:
        self._core.fit(corpus, parallel)

    def add(self, ids: list[str], texts: list[str], parallel: bool = True) -> None:
        # _RelevanceCore.add() doesn't accept parallel (only fit()/search_raw()
        # do) -- passing it here raised TypeError for every caller. `parallel`
        # stays in this method's own signature for interface consistency with
        # fit()/search(), it just isn't forwarded to something that can't take it.
        self._core.add(ids, texts)

    def update(self, ids: list[str], texts: list[str]) -> None:
        self._core.update(ids, texts)

    def delete(self, ids: list[str]) -> None:
        self._core.delete(ids)

    def search(
        self,
        query: str,
        k: int = 10,
        parallel: bool = True,
        candidates: np.ndarray | None = None,
    ) -> list[SearchResult]:
        # _RelevanceCore.search() doesn't accept parallel either (see add()
        # above) -- same fix, same reasoning.
        return self._core.search(query, k, candidates=candidates)

    def search_raw(
        self,
        queries: str | list[str],
        k: int,
        parallel: bool = False,
        candidates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`candidates`, if given, is a single flat array of positions applied
        to every query in this call (matching every other TextIndex backend's
        1D-only candidates contract). Uses bm25s's native `weight_mask` -- a
        post-hoc multiplicative mask zeroing non-candidate scores before
        top-k selection, built once per call and shared across the whole
        batch. `_RelevanceCore` is TAAT/dense-accumulate (same architecture
        as bm25c), so this costs no more accumulation work than an
        unrestricted search -- only the final ranking differs."""
        return self._core.search_raw(queries, k, parallel, candidates=candidates)

    def save(self, directory: str) -> None:
        self._core.save(directory)

    @classmethod
    def load(cls, directory: str) -> RelevanceIndex:
        obj = cls.__new__(cls)
        obj._core = _RelevanceCore.load(directory)
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
        return "relevance"

    @property
    def ids(self) -> list[str]:
        return self._core.ids
