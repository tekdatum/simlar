from __future__ import annotations

from typing import Any

import numpy as np
from simlar_engine import MetadataFilter, SQLFilter

_Filter = str | dict[str, Any] | MetadataFilter

class FilteredIndex:
    """
    Example::

        from simlar import FilteredIndex, HelixIndex

        idx = FilteredIndex(HelixIndex())
        idx.add(
            ids=["a", "b"],
            texts=["apple pie", "car engine"],
            vectors=embeddings,
            metadata=[{"len": 9, "tags": ["food"]}, {"len": 10, "tags": ["cars"]}],
            roles=[["public"], ["staff"]],
        )
        results = idx.search(query_vector=q_vec, k=5, filter="len >= 6 AND 'food' IN tags")
        results = idx.search(query_text="engine", query_vector=q_vec, k=5, roles=["staff"])
        idx.save("filtered.idx")
        idx = FilteredIndex.load("filtered.idx")
    """

    def __init__(self, inner: Any, sql_filter: SQLFilter | None = None) -> None: ...
    def add(
        self,
        ids: list[str],
        *args: Any,
        metadata: list[dict[str, Any]] | None = None,
        roles: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def update_metadata(self, ids: list[str], metadata: list[dict[str, Any]]) -> None: ...
    def grant(self, ids: list[str], roles: list[str]) -> None: ...
    def revoke(self, ids: list[str], roles: list[str]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def candidates(
        self,
        filter: _Filter | None = None,
        roles: list[str] | None = None,
    ) -> np.ndarray | None: ...
    def search(
        self,
        *args: Any,
        filter: _Filter | None = None,
        roles: list[str] | None = None,
        candidates: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any: ...
    def search_raw(
        self,
        *args: Any,
        filter: _Filter | None = None,
        roles: list[str] | None = None,
        candidates: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any: ...
    def save(self, directory: str, base_dir: str | None = None) -> None: ...
    @classmethod
    def load(cls, directory: str, base_dir: str | None = None) -> FilteredIndex: ...
    @property
    def inner(self) -> Any: ...
    @property
    def sql_filter(self) -> SQLFilter: ...
    @property
    def index_type(self) -> str: ...
    @property
    def size(self) -> int: ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def ids(self) -> list[str]: ...
    def __getattr__(self, name: str) -> Any: ...
