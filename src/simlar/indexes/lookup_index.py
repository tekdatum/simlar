from __future__ import annotations

import numpy as np
from simlar_engine.indexes._lookup_impl import _TextCore

from simlar.contracts import SearchResult, TextIndex
from simlar.indexes.registry import register


@register("lookup")
class LookupIndex(TextIndex):
    """
    Example:
        >>> idx = LookupIndex()
        >>> idx.add(["a", "b"], ["hello world", "foo bar"])
        >>> results = idx.search("hello", k=5)
        >>> results[0].id
        'a'
    """

    def __init__(
        self,
        stopwords_lang: str = "english",
        stemmer_lang: str = "english",
    ) -> None:
        self._core = _TextCore(stopwords_lang, stemmer_lang)

    # ── Public contract ───────────────────────────────────────────────────────

    def fit(self, corpus: list[str], parallel: bool = False, **kwargs: object) -> None:
        self._core.fit(corpus, parallel)

    def add(self, ids: list[str], texts: list[str]) -> None:
        self._core.add(ids, texts)

    def update(self, ids: list[str], texts: list[str]) -> None:
        self._core.update(ids, texts)

    def delete(self, ids: list[str]) -> None:
        self._core.delete(ids)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        return self._core.search(query, k)

    def search_raw(
        self,
        queries: str | list[str],
        k: int,
        parallel: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._core.search_raw(queries, k, parallel)

    def save(self, directory: str) -> None:
        self._core.save(directory)

    @classmethod
    def load(cls, directory: str) -> LookupIndex:
        obj = cls.__new__(cls)
        obj._core = _TextCore.load(directory)
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
        return self._core.index_type
