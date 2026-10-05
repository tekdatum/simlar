from __future__ import annotations

from simlar._engine import engine_class, engine_core
from simlar.indexes.registry import load_from_directory, register

# The engine's filtering types. On an engine build without filtering these are stand-ins that
# raise IndexUnavailableError when used; FilterError stays a real exception so `except` works.
_EngineFilteredIndex = engine_class("filtered", "simlar_engine", "FilteredIndex")
SQLFilter = engine_class("filtered", "simlar_engine", "SQLFilter")
MetadataFilter = engine_class("filtered", "simlar_engine", "MetadataFilter")
FilterError = engine_core("filtered", "simlar_engine", "FilterError")
if FilterError is None:

    class FilterError(ValueError):
        """Stand-in: this engine build has no filtering, so nothing raises it."""


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
