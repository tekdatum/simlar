"""LlamaIndex vector store backed by simlar's FilteredIndex over a HelixIndex.

Plugs into the standard ``VectorStoreIndex`` pipeline with LlamaIndex's
``MetadataFilters`` and query modes, following the same conventions as
``PGVectorStore``:

* ``vector_store_query_mode="default"`` -- vector search
* ``"sparse"`` / ``"text_search"`` -- BM25 text search
* ``"hybrid"`` -- BM25 + vector, fused with RRF

Filters are resolved by the engine's SQL metadata filter to the allowed
documents *before* ranking, so results are the exact top-k among them.
Every write goes straight to the index (append, in-place update, compacting
delete) -- nothing is rebuilt and no copy of the embeddings is kept.

Example::

    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    vector_store = SimlarVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    retriever = index.as_retriever(
        similarity_top_k=5,
        vector_store_query_mode="hybrid",
        filters=MetadataFilters(filters=[MetadataFilter(key="year", value=2020, operator=">=")]),
        vector_store_kwargs={"roles": ["staff"]},
    )
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

from simlar.indexes.filtered_index import FilteredIndex
from simlar.indexes.helix_index import HelixIndex
from simlar.integrations._filtering import filterable_metadata, replacement_patch

_INDEX_DIRNAME = "index"
_DOCS_FILENAME = "docs.json"
_DEFAULT_VECTOR_STORE = "default"
_NAMESPACE_SEP = "__"
_DEFAULT_PERSIST_FNAME = "vector_store.json"

# Conditions that never / always hold: `position` is a builtin column, >= 0.
_NEVER: dict = {"position": {"$lt": 0}}
_ALWAYS: dict = {"position": {"$gte": 0}}

_COMPARISONS = {
    FilterOperator.EQ: "$eq",
    FilterOperator.NE: "$ne",
    FilterOperator.GT: "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.IN: "$in",
    FilterOperator.NIN: "$nin",
}


# ── MetadataFilters -> engine filter ──────────────────────────────────────────


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _condition(f: MetadataFilter, known: set[str]) -> dict:
    op = getattr(f, "operator", FilterOperator.EQ)  # legacy ExactMatchFilter has none
    if f.key.lower() not in known:
        # Like PGVectorStore: a key no document has matches nothing (a
        # missing value is NULL), except IS_EMPTY, which it satisfies.
        return _ALWAYS if op == FilterOperator.IS_EMPTY else _NEVER
    if op in _COMPARISONS:
        value = _as_list(f.value) if op in (FilterOperator.IN, FilterOperator.NIN) else f.value
        return {f.key: {_COMPARISONS[op]: value}}
    if op == FilterOperator.CONTAINS:
        return {f.key: {"$contains": f.value}}
    if op == FilterOperator.ANY:
        # On a multi-valued key IN is true if any of the doc's values is listed.
        return {f.key: {"$in": _as_list(f.value)}}
    if op == FilterOperator.ALL:
        return {"$and": [{f.key: {"$contains": v}} for v in _as_list(f.value)]}
    if op in (FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE):
        return {f.key: {"$like": f"%{f.value}%"}}
    if op == FilterOperator.IS_EMPTY:
        return {f.key: {"$exists": False}}
    raise NotImplementedError(f"Filter operator {op!r} is not supported by SimlarVectorStore.")


def to_engine_filter(filters: MetadataFilters | None, known: set[str]) -> dict | None:
    """Translate LlamaIndex ``MetadataFilters`` into the engine's filter dict.

    `known` is the lower-cased set of filterable keys present in the store.
    Nested ``MetadataFilters`` recurse; ``NOT`` means none of the sub-filters
    match (as in LlamaIndex's own filter semantics). Returns None for an
    empty filter.
    """
    if filters is None or not filters.filters:
        return None
    parts = [
        to_engine_filter(f, known) if isinstance(f, MetadataFilters) else _condition(f, known)
        for f in filters.filters
    ]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    condition = filters.condition or FilterCondition.AND
    if condition == FilterCondition.NOT:
        return {"$not": parts[0] if len(parts) == 1 else {"$or": parts}}
    if len(parts) == 1:
        return parts[0]
    return {"$and" if condition == FilterCondition.AND else "$or": parts}


def _all_of(*filters: dict | None) -> dict | None:
    parts = [f for f in filters if f is not None]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


# ── Vector store ──────────────────────────────────────────────────────────────


class SimlarVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store with hybrid search and SQL metadata filtering.

    Args:
        text_k: Text candidate pool size fed into RRF.
        vector_k: Vector candidate pool size fed into RRF.
        top_k: Default result list length of the HelixIndex.
        parallel: Default threading mode for index writes and queries.
            Override per call with ``parallel=`` (``vector_store_kwargs`` on
            a retriever).

    Node metadata keys that are identifiers with scalar (or list-of-scalar)
    values are filterable, plus ``ref_doc_id`` / ``doc_id`` /
    ``document_id``. ``vector_store_kwargs={"roles": [...]}`` restricts
    results to nodes granted to any of those roles (``roles=`` on ``add``).
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    text_k: int = 500
    vector_k: int = 200
    top_k: int = 100
    parallel: bool = True

    _index: FilteredIndex = PrivateAttr()
    # node id -> (text, node_to_metadata_dict payload)
    _docs: dict[str, tuple[str, dict]] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        *,
        text_k: int = 500,
        vector_k: int = 200,
        top_k: int = 100,
        parallel: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]  # pydantic fields
            text_k=text_k, vector_k=vector_k, top_k=top_k, parallel=parallel, **kwargs
        )
        self._index = self._new_index()

    @classmethod
    def class_name(cls) -> str:
        return "SimlarVectorStore"

    def _new_index(self) -> FilteredIndex:
        return FilteredIndex(
            HelixIndex(text_k=self.text_k, vector_k=self.vector_k, top_k=self.top_k)
        )

    @property
    def client(self) -> FilteredIndex:
        """The underlying ``FilteredIndex(HelixIndex)``."""
        return self._index

    # ── Writes ────────────────────────────────────────────────────────────────

    @staticmethod
    def _filterable(payload: dict) -> dict:
        return filterable_metadata(
            {k: v for k, v in payload.items() if k not in ("_node_content", "_node_type")}
        )

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """Add nodes, or replace the ones whose id is already stored.

        Nodes must have embeddings set (``VectorStoreIndex`` does this). Only
        these nodes are indexed: new ids are appended, existing ids updated
        in place.

        Args:
            nodes: Nodes with embeddings.
            roles: Optional list of role lists, one per node, granting those
                roles access for ``roles=`` queries (additive).
            parallel: Thread this write. Defaults to the store setting.
        """
        roles = add_kwargs.get("roles")
        parallel = add_kwargs.get("parallel", self.parallel)
        n = len(nodes)
        if roles is not None and len(roles) != n:
            raise ValueError(f"roles length {len(roles)} != nodes length {n}")
        if n == 0:
            return []
        embeddings = [
            node.get_embedding() if node.embedding is not None else None for node in nodes
        ]
        if any(e is None for e in embeddings):
            raise ValueError(
                "All nodes must have embeddings set before adding to SimlarVectorStore."
            )

        ids = [node.node_id for node in nodes]
        texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        payloads = [
            node_to_metadata_dict(node, remove_text=True, flat_metadata=False) for node in nodes
        ]
        fmetas = [self._filterable(p) for p in payloads]
        vectors = np.asarray(embeddings, dtype=np.float32)

        # A repeated id keeps its last occurrence.
        order = list({id_: i for i, id_ in enumerate(ids)}.values())
        new = [i for i in order if ids[i] not in self._docs]
        old = [i for i in order if ids[i] in self._docs]
        if new:
            self._index.add(
                [ids[i] for i in new],
                [texts[i] for i in new],
                vectors if len(new) == n else vectors[new],
                metadata=[fmetas[i] for i in new],
                roles=None if roles is None else [roles[i] for i in new],
                parallel=parallel,
            )
        if old:
            self._index.update(
                [ids[i] for i in old],
                [texts[i] for i in old],
                vectors if len(old) == n else vectors[old],
                metadata=[
                    replacement_patch(self._filterable(self._docs[ids[i]][1]), fmetas[i])
                    for i in old
                ],
            )
            for i in old:
                if roles is not None and roles[i]:
                    self._index.grant([ids[i]], list(roles[i]))

        for i in order:
            self._docs[ids[i]] = (texts[i], payloads[i])
        return ids

    def _delete_ids(self, ids: list[str]) -> None:
        doomed = [i for i in dict.fromkeys(ids) if i in self._docs]
        if not doomed:
            return
        if len(doomed) == len(self._docs):
            # The BM25 core refuses to delete its last document; start fresh.
            self.clear()
            return
        self._index.delete(doomed)
        for id_ in doomed:
            del self._docs[id_]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete every node of source document ``ref_doc_id``."""
        self._delete_ids(self._matching_ids({"ref_doc_id": ref_doc_id}))

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by id and/or metadata filter (both: AND). No-op with neither."""
        if not node_ids and not filters:
            return
        self._delete_ids(self._matching_ids(self._filter(filters, node_ids=node_ids)))

    def clear(self) -> None:
        """Delete every node."""
        self._docs.clear()
        self._index = self._new_index()

    def grant(self, node_ids: list[str], roles: list[str]) -> None:
        """Let nodes ``node_ids`` be returned to queries made with any of ``roles``."""
        self._index.grant(node_ids, roles)

    def revoke(self, node_ids: list[str], roles: list[str]) -> None:
        """Remove ``roles``' access to nodes ``node_ids``."""
        self._index.revoke(node_ids, roles)

    # ── Reads ─────────────────────────────────────────────────────────────────

    def _known_keys(self) -> set[str]:
        scalar, multi = self._index.sql_filter.schema()
        return {k.lower() for k in scalar | multi} | {"id", "position"}

    def _filter(
        self,
        filters: MetadataFilters | None,
        node_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> dict | None:
        return _all_of(
            to_engine_filter(filters, self._known_keys()) if filters else None,
            {"id": {"$in": list(node_ids)}} if node_ids else None,
            {"ref_doc_id": {"$in": list(doc_ids)}} if doc_ids else None,
        )

    def _matching_ids(self, filter: dict | None) -> list[str]:
        if not self._docs:
            return []
        ids = self._index.ids
        positions = self._index.candidates(filter)
        return ids if positions is None else [ids[p] for p in positions]

    def _node(self, id_: str) -> BaseNode:
        text, payload = self._docs[id_]
        node = metadata_dict_to_node(payload)
        node.set_content(text)
        return node

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        """Nodes matching ``node_ids`` and/or ``filters`` (both: AND)."""
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided.")
        return [self._node(i) for i in self._matching_ids(self._filter(filters, node_ids=node_ids))]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Search with ``query.mode``: ``DEFAULT`` (vector), ``SPARSE`` /
        ``TEXT_SEARCH`` (BM25) or ``HYBRID`` (both, RRF-fused).

        ``query.filters``, ``query.doc_ids`` and ``query.node_ids`` restrict
        the candidates before ranking. Accepts ``roles`` and ``parallel``
        keyword arguments (``vector_store_kwargs`` on a retriever).
        """
        mode = query.mode
        query_text: str | None = None
        query_vector: Any = None
        if mode == VectorStoreQueryMode.DEFAULT:
            query_vector = query.query_embedding
            k = query.similarity_top_k
        elif mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH):
            query_text = query.query_str
            k = query.sparse_top_k or query.similarity_top_k
        elif mode == VectorStoreQueryMode.HYBRID:
            query_text, query_vector = query.query_str, query.query_embedding
            k = query.hybrid_top_k or query.similarity_top_k
        else:
            raise NotImplementedError(f"Query mode {mode!r} is not supported by SimlarVectorStore.")
        if mode != VectorStoreQueryMode.DEFAULT and not query_text:
            raise ValueError(f"query.query_str is required for {mode.value!r} mode.")
        needs_vector = mode in (VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID)
        if needs_vector and query_vector is None:
            raise ValueError(f"query.query_embedding is required for {mode.value!r} mode.")

        if not self._docs or k <= 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        parallel = kwargs.get("parallel")
        results = self._index.search(
            query_text=query_text,
            query_vector=None
            if query_vector is None
            else np.asarray(query_vector, dtype=np.float32),
            k=min(k, len(self._docs)),
            parallel=self.parallel if parallel is None else parallel,
            filter=self._filter(query.filters, node_ids=query.node_ids, doc_ids=query.doc_ids),
            roles=kwargs.get("roles"),
        )
        return VectorStoreQueryResult(
            nodes=[self._node(r.id) for r in results],
            similarities=[float(r.score) for r in results],
            ids=[r.id for r in results],
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def persist(self, persist_path: str, fs: Any | None = None) -> None:
        """Save to directory ``persist_path``: ``index/`` (FilteredIndex) and
        ``docs.json`` (texts, node payloads, settings). Local filesystem only."""
        _assert_local_fs(fs)
        d = Path(persist_path)
        d.mkdir(parents=True, exist_ok=True)
        if self._docs:
            self._index.save(str(d / _INDEX_DIRNAME))
        payload = {
            "text_k": self.text_k,
            "vector_k": self.vector_k,
            "top_k": self.top_k,
            "parallel": self.parallel,
            # In index order, so loading can check alignment.
            "docs": [[id_, *self._docs[id_]] for id_ in (self._index.ids if self._docs else [])],
        }
        (d / _DOCS_FILENAME).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_persist_path(
        cls, persist_path: str, fs: Any | None = None, **kwargs: Any
    ) -> SimlarVectorStore:
        """Load a store written by ``persist``. ``parallel=`` overrides the saved setting."""
        _assert_local_fs(fs)
        d = Path(persist_path)
        if not (d / _DOCS_FILENAME).exists():
            raise ValueError(f"No saved SimlarVectorStore found at {persist_path!r}.")
        payload = json.loads((d / _DOCS_FILENAME).read_text(encoding="utf-8"))
        store = cls(
            text_k=payload["text_k"],
            vector_k=payload["vector_k"],
            top_k=payload["top_k"],
            parallel=kwargs.get("parallel", payload["parallel"]),
        )
        store._docs = {id_: (text, meta) for id_, text, meta in payload["docs"]}
        if store._docs:
            store._index = FilteredIndex.load(str(d / _INDEX_DIRNAME))
            if store._index.ids != list(store._docs):
                raise ValueError(f"Corrupted store at {persist_path!r}: index and nodes disagree.")
        return store

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        namespace: str = _DEFAULT_VECTOR_STORE,
        fs: Any | None = None,
        **kwargs: Any,
    ) -> SimlarVectorStore:
        """Load from the path ``StorageContext.persist(persist_dir)`` wrote to."""
        path = os.path.join(persist_dir, f"{namespace}{_NAMESPACE_SEP}{_DEFAULT_PERSIST_FNAME}")
        return cls.from_persist_path(path, fs=fs, **kwargs)


def _assert_local_fs(fs: Any) -> None:
    """Raise NotImplementedError if ``fs`` is a non-local fsspec filesystem."""
    if fs is None:
        return
    from fsspec.implementations.local import LocalFileSystem

    if not isinstance(fs, LocalFileSystem):
        raise NotImplementedError("SimlarVectorStore only supports local storage.")
