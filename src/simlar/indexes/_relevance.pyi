from __future__ import annotations

import numpy as np

from simlar.contracts import SearchResult, TextIndex

class RelevanceIndex(TextIndex):
    """

    Example::

        idx = RelevanceIndex(k1=1.5, b=0.75)
        idx.add(ids=["a", "b"], texts=["first doc", "second doc"])
        results = idx.search("first", k=10)
        idx.save("bm25.idx")
        idx = RelevanceIndex.load("idx")
    """

    def __init__(
        self,
        method: str = "robertson",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords_lang: str = "english",
        stemmer_lang: str = "english",
    ) -> None: ...
    def fit(self, corpus: list[str], parallel: bool = False, **kwargs: object) -> None: ...
    def add(self, ids: list[str], texts: list[str], parallel: bool = False) -> None: ...
    def update(self, ids: list[str], texts: list[str]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search(self, query: str, k: int, parallel: bool = False) -> list[SearchResult]: ...
    def search_raw(
        self,
        queries: str | list[str],
        k: int,
        parallel: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def save(self, directory: str) -> None: ...
    @classmethod
    def load(cls, directory: str) -> RelevanceIndex: ...
    @property
    def size(self) -> int: ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def index_type(self) -> str: ...
    @property
    def ids(self) -> list[str]: ...
