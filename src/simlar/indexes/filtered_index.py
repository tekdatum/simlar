from __future__ import annotations

from simlar_engine import FilteredIndex as _EngineFilteredIndex

from simlar.indexes.registry import load_from_directory, register


@register("filtered")
class FilteredIndex(_EngineFilteredIndex):
    """SQL metadata filtering and role-based access in front of any index.

    Wraps any simlar index -- ``HelixIndex``, ``SimlarEngine``,
    ``RelevanceIndex``, ``LookupIndex``, ``BM25CIndex`` or
    ``StreamingHybridIndex`` -- plus a SQLite metadata store. ``filter=``
    and ``roles=`` are resolved to the allowed positions first and passed
    to the inner index as ``candidates=``, so results are the exact top-k
    among the allowed documents.

    Example::

        idx = FilteredIndex(HelixIndex())
        idx.add(
            ["a", "b"], ["apple pie", "car engine"], vectors,
            metadata=[{"len": 9, "tags": ["food"]}, {"len": 10, "tags": ["cars"]}],
            roles=[["public"], ["staff"]],
        )
        idx.search(query_vector=q, k=5, filter="len >= 6 AND 'food' IN tags")
        idx.search(query_text="engine", query_vector=q, k=5, roles=["staff"])

    ``filter`` is a SQL WHERE fragment (``"len >= 6 AND lang IN ('en','es')"``),
    a LangChain-style dict (``{"len": {"$gte": 6}}``) or a
    ``SQLFilter.where(...)`` chain. ``roles=None`` skips the access check.
    """

    # Load the inner index back as the simlar wrapper it was saved from.
    _load_inner = staticmethod(load_from_directory)

    @classmethod
    def load(cls, directory: str, base_dir=None, *, embedder=None) -> FilteredIndex:
        obj = super().load(directory) if base_dir is None else super().load(directory, base_dir)
        if embedder is not None:
            obj.embedder = embedder
        return obj

    # The inner index owns the embedder; __getattr__ only forwards reads.
    @property
    def embedder(self):
        return getattr(self.inner, "embedder", None)

    @embedder.setter
    def embedder(self, value) -> None:
        if not hasattr(self.inner, "embedder"):
            raise TypeError(f"{type(self.inner).__name__} does not take an embedder")
        self.inner.embedder = value
