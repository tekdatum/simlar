from __future__ import annotations

import numpy as np
from simlar_engine.indexes._hash_match_impl import _HashMatchCore

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
    ) -> None:
        self._core = _HashMatchCore(stopwords_lang)

    # ── Public contract ───────────────────────────────────────────────────────

    def fit(self, corpus: list[str], parallel: bool = True, **kwargs: object) -> None:
        self._core.fit(corpus, parallel)

    def add(self, ids: list[str], texts: list[str], parallel: bool = True) -> None:
        self._core.add(ids, texts, parallel)

    def update(self, ids: list[str], texts: list[str]) -> None:
        self._core.update(ids, texts)

    def delete(self, ids: list[str]) -> None:
        self._core.delete(ids)

    def search(
        self,
        query: str | list[str],
        k: int = 10,
        parallel: bool = True,
        batch_size: int | None = None,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Rank documents against `query`, or against a batch of queries.

        A single string returns `list[SearchResult]`; a list of strings returns
        one such list per query, in order. A batch runs as one threaded call
        rather than one call per query, which is what `parallel` acts on.
        """
        return self._core.search(query, k, parallel, batch_size)

    def search_raw(
        self,
        queries: str | list[str],
        k: int,
        parallel: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._core.search_raw(queries, k, parallel)

    def save(self, directory: str, base_dir: str | None = None) -> None:
        self._core.save(directory, base_dir)

    @classmethod
    def load(cls, directory: str, base_dir: str | None = None) -> LookupIndex:
        obj = cls.__new__(cls)
        obj._core = _HashMatchCore.load(directory, base_dir)
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

    @property
    def ids(self) -> list[str]:
        return self._core.ids
