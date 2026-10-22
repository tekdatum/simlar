"""SimlarDocumentStore -- a Haystack 3.x DocumentStore on simlar's FilteredIndex.

Hybrid (BM25 + vector) retrieval over a ``FilteredIndex(HelixIndex)``:
Haystack ``filters`` (https://docs.haystack.deepset.ai/docs/metadata-filtering)
and ``roles`` are resolved in SQL to the allowed positions first and passed to
the index as ``candidates=``, so results are the exact top-k among the allowed
documents. Writes go straight to the index -- new ids are appended,
overwritten ids are updated in place, deletes compact -- so nothing is ever
rebuilt or tombstoned.

The store does not own an embedder: documents must arrive with ``embedding``
set by an upstream Haystack embedder, and queries bring their own embedding.

Example::

    store = SimlarDocumentStore()
    store.write_documents(embedded_docs, roles=[["public"]] * len(embedded_docs))
    store.hybrid_retrieval(
        "query", query_embedding,
        filters={"field": "meta.year", "operator": ">=", "value": 2020},
        roles=["public"],
    )
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from haystack import Document, default_from_dict, default_to_dict, logging
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.filters import document_matches_filter

from simlar.indexes.filtered_index import FilteredIndex
from simlar.indexes.helix_index import HelixIndex
from simlar.integrations.haystack._filters import MetaFields, compile_filters

logger = logging.getLogger(__name__)

_INDEX_DIRNAME = "index"
_STORE_FILENAME = "store.json"
_FORMAT_VERSION = 2

_Mode = Literal["embedding", "bm25", "hybrid"]


class SimlarDocumentStore:
    """Haystack DocumentStore backed by a simlar ``FilteredIndex(HelixIndex)``.

    Filters follow Haystack's filter syntax and semantics exactly. Conditions
    SQL can evaluate faithfully run as one cached SQL query; the rest (ISO-date
    string comparisons, nested or non-scalar metadata, ``content``) narrow the
    candidates in SQL and are then checked with Haystack's own
    ``document_matches_filter``.

    ``roles`` restricts results to documents granted to any of those roles
    (see ``write_documents(roles=...)``, ``grant``, ``revoke``); ``None``
    skips the access check.
    """

    def __init__(
        self,
        top_k: int = 5,
        relevance_k: int = 100,
        core_k: int = 50,
        parallel: bool = True,
    ):
        """
        Args:
            top_k: Default number of documents returned by :meth:`search`.
            relevance_k: Text (BM25) candidate pool size fed into hybrid fusion.
            core_k: Vector candidate pool size fed into hybrid fusion.
            parallel: Default threading mode for writes and searches. Override
                per call with ``parallel=``.
        """
        self._top_k = top_k
        self._relevance_k = relevance_k
        self._core_k = core_k
        self._parallel = parallel
        self._reset()

    def _reset(self) -> None:
        self._index = FilteredIndex(
            HelixIndex(text_k=self._relevance_k, vector_k=self._core_k, top_k=self._top_k)
        )
        # _docs[p] is the document at index position p.
        self._docs: list[Document] = []
        self._pos: dict[str, int] = {}
        self._fields = MetaFields()
        self._dim: int | None = None

    @property
    def index(self) -> FilteredIndex:
        """The underlying ``FilteredIndex(HelixIndex)``."""
        return self._index

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            top_k=self._top_k,
            relevance_k=self._relevance_k,
            core_k=self._core_k,
            parallel=self._parallel,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimlarDocumentStore:
        return default_from_dict(cls, data)

    # ── Writes ────────────────────────────────────────────────────────────────

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        roles: Sequence[Sequence[str]] | None = None,
        parallel: bool | None = None,
    ) -> int:
        """Write documents to the store.

        Args:
            documents: Documents with their ``embedding`` already set.
            policy: What to do with ids already stored. ``NONE`` overwrites,
                as ``OVERWRITE`` does: the document is replaced in place.
            roles: Optional list of role lists, one per document, granting
                those roles access for ``roles=`` retrieval (additive).
            parallel: Thread the index write. Defaults to the store setting.

        Returns:
            The number of documents written.
        """
        if not isinstance(documents, (list, tuple)) or any(
            not isinstance(d, Document) for d in documents
        ):
            raise ValueError("Please provide a list of Documents.")
        if roles is not None and len(roles) != len(documents):
            raise ValueError(f"roles length {len(roles)} != documents length {len(documents)}")

        # Resolve duplicates up front, so a FAIL raises before anything is written.
        chosen: dict[str, int] = {}
        for i, doc in enumerate(documents):
            if doc.id in self._pos or doc.id in chosen:
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"ID '{doc.id}' already exists.")
                if policy == DuplicatePolicy.SKIP:
                    logger.warning("ID '{document_id}' already exists", document_id=doc.id)
                    continue
            chosen[doc.id] = i  # OVERWRITE / NONE: the last occurrence wins
        # Per the protocol, only SKIP reports fewer documents than it was given.
        written = len(chosen) if policy == DuplicatePolicy.SKIP else len(documents)
        if not chosen:
            return written

        order = list(chosen.values())
        docs = [documents[i] for i in order]
        vectors = self._vectors(docs)
        # Stored copies own their meta dict, so a caller mutating the documents
        # it passed in can't drift from the metadata indexed in SQL.
        stored = [replace(d, meta=dict(d.meta)) for d in docs]
        parallel = self._parallel if parallel is None else parallel
        new = [j for j, d in enumerate(docs) if d.id not in self._pos]
        old = [j for j, d in enumerate(docs) if d.id in self._pos]

        fields_before = copy.deepcopy(self._fields)
        try:
            if new:
                self._index.add(
                    [docs[j].id for j in new],
                    [docs[j].content or "" for j in new],
                    vectors if len(new) == len(docs) else vectors[new],
                    metadata=[self._fields.sql_metadata(docs[j].meta) for j in new],
                    roles=None if roles is None else [list(roles[order[j]]) for j in new],
                    parallel=parallel,
                )
                for j in new:
                    self._pos[docs[j].id] = len(self._docs)
                    self._docs.append(stored[j])
            if old:
                self._index.update(
                    [docs[j].id for j in old],
                    [docs[j].content or "" for j in old],
                    vectors if len(old) == len(docs) else vectors[old],
                    metadata=[self._replacement(docs[j]) for j in old],
                )
                for j in old:
                    self._docs[self._pos[docs[j].id]] = stored[j]
                    if roles is not None and roles[order[j]]:
                        self._index.grant([docs[j].id], list(roles[order[j]]))
        except BaseException:
            self._fields = fields_before
            raise
        if self._dim is None:
            self._dim = vectors.shape[1]
        return written

    def _vectors(self, docs: list[Document]) -> np.ndarray:
        missing = [d.id for d in docs if d.embedding is None]
        if missing:
            raise ValueError(
                f"Documents {missing} have no embedding. "
                "Run a Haystack embedder component before writing to SimlarDocumentStore."
            )
        try:
            vectors = np.asarray([d.embedding for d in docs], dtype=np.float32)
        except ValueError:
            raise ValueError("All document embeddings must have the same dimension.") from None
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("Document embeddings must be non-empty lists of floats.")
        if self._dim is not None and vectors.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dimension {vectors.shape[1]} does not match the store's {self._dim}."
            )
        return vectors

    def _replacement(self, doc: Document) -> dict[str, Any]:
        """SQL metadata patch replacing the stored doc's metadata with `doc`'s:
        keys the new metadata drops are cleared."""
        old = self._docs[self._pos[doc.id]]
        patch = dict.fromkeys(self._fields.sql_metadata(old.meta))
        patch.update(self._fields.sql_metadata(doc.meta))
        return patch

    def delete_documents(self, document_ids: list[str]) -> None:
        """Delete documents by id. Unknown ids are ignored."""
        doomed = [i for i in dict.fromkeys(document_ids) if i in self._pos]
        if not doomed:
            return
        if len(doomed) == len(self._docs):
            # The BM25 core refuses to delete its last document; start fresh.
            self.delete_all_documents()
            return
        self._index.delete(doomed)
        gone = set(doomed)
        self._docs = [d for d in self._docs if d.id not in gone]
        self._pos = {d.id: p for p, d in enumerate(self._docs)}

    def delete_all_documents(self) -> None:
        """Delete every document."""
        self._reset()

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """Delete all documents matching `filters`. Returns the number deleted."""
        ids = [d.id for d in self._matching(filters)]
        self.delete_documents(ids)
        return len(ids)

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """Merge `meta` into the metadata of all documents matching `filters`.
        Returns the number updated."""
        allowed = self._allowed(filters, None)
        positions = range(len(self._docs)) if allowed is None else allowed.tolist()
        if not positions:
            return 0
        patch = self._fields.sql_metadata(meta)
        ids = []
        for p in positions:
            doc = self._docs[p]
            self._docs[p] = replace(doc, meta={**doc.meta, **meta})
            ids.append(doc.id)
        if patch:
            self._index.update_metadata(ids, [patch] * len(ids))
        return len(ids)

    def grant(self, document_ids: list[str], roles: list[str]) -> None:
        """Let documents be returned to retrievals made with any of `roles`."""
        self._index.grant(document_ids, roles)

    def revoke(self, document_ids: list[str], roles: list[str]) -> None:
        """Remove `roles`' access to documents."""
        self._index.revoke(document_ids, roles)

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _allowed(
        self, filters: dict[str, Any] | None, roles: Sequence[str] | None
    ) -> np.ndarray | None:
        """Sorted positions matching `filters` and `roles`; None means all."""
        metadata_filter, exact = compile_filters(filters, self._fields) if filters else (None, True)
        if metadata_filter is None and roles is None:
            positions = None
        else:
            positions = self._index.candidates(
                metadata_filter, None if roles is None else list(roles)
            )
        if exact:
            return positions
        candidates = range(len(self._docs)) if positions is None else positions.tolist()
        return np.fromiter(
            (p for p in candidates if document_matches_filter(filters, self._docs[p])),  # type: ignore[arg-type]
            dtype=np.int64,
        )

    def _matching(
        self, filters: dict[str, Any] | None, roles: Sequence[str] | None = None
    ) -> list[Document]:
        positions = self._allowed(filters, roles)
        if positions is None:
            return list(self._docs)
        return [self._docs[p] for p in positions.tolist()]

    def filter_documents(
        self, filters: dict[str, Any] | None = None, roles: Sequence[str] | None = None
    ) -> list[Document]:
        """Documents matching `filters` (Haystack filter syntax) and visible to
        `roles`, in insertion order."""
        return [replace(d, meta=dict(d.meta)) for d in self._matching(filters, roles)]

    def count_documents(self) -> int:
        return len(self._docs)

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        positions = self._allowed(filters, None)
        return len(self._docs) if positions is None else int(positions.size)

    # ── Metadata introspection ────────────────────────────────────────────────

    def count_unique_metadata_by_filter(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """Number of unique values per metadata field among documents matching
        `filters`. Field names may include the ``meta.`` prefix."""
        docs = self._matching(filters)
        result: dict[str, int] = {}
        for name in metadata_fields:
            key = name.removeprefix("meta.")
            result[key] = len({_hashable(d.meta[key]) for d in docs if d.meta.get(key) is not None})
        return result

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """Metadata fields with types inferred from stored values
        (``keyword``, ``int``, ``float``, ``boolean``)."""
        types: dict[str, str] = {}
        for doc in self._docs:
            for key, value in doc.meta.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    types[key] = "boolean"
                elif isinstance(value, int):
                    types[key] = "int"
                elif isinstance(value, float):
                    types[key] = "float"
                else:
                    types[key] = "keyword"
        return {key: {"type": t} for key, t in types.items()}

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, Any]:
        """Min and max of a metadata field across all documents; both None if
        the field has no comparable values."""
        key = metadata_field.removeprefix("meta.")
        values = [d.meta[key] for d in self._docs if isinstance(d.meta.get(key), (int, float, str))]
        try:
            return (
                {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}
            )
        except TypeError:
            return {"min": None, "max": None}

    def get_metadata_field_unique_values(
        self,
        metadata_field: str,
        search_term: str | None = None,
        from_: int = 0,
        size: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[Any], int]:
        """A page of the unique values of a metadata field, and the total number
        of unique values. `search_term` keeps values containing it
        (case-insensitive); `filters` restricts the documents considered."""
        key = metadata_field.removeprefix("meta.")
        unique: dict[tuple[str, Any], Any] = {}
        for doc in self._matching(filters):
            value = doc.meta.get(key)
            if value is not None:
                unique.setdefault((type(value).__name__, _hashable(value)), value)
        if search_term:
            term = search_term.lower()
            unique = {k: v for k, v in unique.items() if term in str(v).lower()}
        ordered = sorted(unique, key=lambda k: (str(unique[k]), k[0]))
        return [unique[k] for k in ordered[from_ : from_ + size]], len(ordered)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def embedding_retrieval(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        roles: Sequence[str] | None = None,
        return_embedding: bool = False,
        parallel: bool | None = None,
    ) -> list[Document]:
        """Vector search, restricted to documents matching `filters` and `roles`.
        Scores are similarities, higher is better."""
        return self._retrieve(
            "embedding", None, query_embedding, filters, top_k, roles, return_embedding, parallel
        )

    def bm25_retrieval(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        roles: Sequence[str] | None = None,
        return_embedding: bool = False,
        parallel: bool | None = None,
    ) -> list[Document]:
        """BM25 keyword search, restricted to documents matching `filters` and
        `roles`. Documents scoring 0 (no query term) are not returned."""
        return self._retrieve(
            "bm25", query, None, filters, top_k, roles, return_embedding, parallel
        )

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        roles: Sequence[str] | None = None,
        return_embedding: bool = False,
        parallel: bool | None = None,
    ) -> list[Document]:
        """BM25 + vector search fused with RRF, restricted to documents matching
        `filters` and `roles`. Scores are fused, higher is better."""
        return self._retrieve(
            "hybrid", query, query_embedding, filters, top_k, roles, return_embedding, parallel
        )

    def search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        parallel: bool | None = None,
    ) -> list[Document]:
        """Hybrid search with the store-level ``top_k`` default; see
        :meth:`hybrid_retrieval`."""
        return self.hybrid_retrieval(
            query_text,
            query_embedding,
            filters=filters,
            top_k=top_k or self._top_k,
            parallel=parallel,
        )

    def _retrieve(
        self,
        mode: _Mode,
        query: str | None,
        query_embedding: list[float] | None,
        filters: dict[str, Any] | None,
        top_k: int,
        roles: Sequence[str] | None,
        return_embedding: bool,
        parallel: bool | None,
    ) -> list[Document]:
        if top_k <= 0 or not self._docs:
            return []
        allowed = self._allowed(filters, roles)
        n = len(self._docs) if allowed is None else int(allowed.size)
        if n == 0:
            return []
        kwargs: dict[str, Any] = {
            "k": min(top_k, n),
            "parallel": self._parallel if parallel is None else parallel,
        }
        if allowed is not None:
            kwargs["candidates"] = allowed
        vector = None if query_embedding is None else np.asarray(query_embedding, dtype=np.float32)

        if mode == "embedding":
            results = self._index.vector_index.search(vector, **kwargs)
        elif mode == "bm25":
            results = [r for r in self._index.text_index.search(query, **kwargs) if r.score > 0]
        else:
            results = self._index.search(query_text=query, query_vector=vector, **kwargs)

        out = []
        for r in results:
            doc = self._docs[self._pos[r.id]]
            out.append(
                replace(
                    doc,
                    meta=dict(doc.meta),
                    score=float(r.score),
                    embedding=doc.embedding if return_embedding else None,
                )
            )
        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the store to a directory.

        Layout::

            <path>/
                index/      <- FilteredIndex: HelixIndex + SQLFilter (metadata, roles)
                store.json  <- documents (in index order) and store settings

        Metadata must be JSON-serializable.
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        if self._docs:
            self._index.save(str(root / _INDEX_DIRNAME))
        payload = {
            "format_version": _FORMAT_VERSION,
            "init_parameters": self.to_dict()["init_parameters"],
            "documents": [d.to_dict(flatten=False) for d in self._docs],
            "fields": {
                "kinds": {k: sorted(v) for k, v in self._fields.kinds.items()},
                "columns": self._fields.columns,
            },
        }
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except TypeError as e:
            raise TypeError(
                f"SimlarDocumentStore metadata must be JSON-serializable: {e}"
            ) from None
        (root / _STORE_FILENAME).write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SimlarDocumentStore:
        """Load a store saved with :meth:`save`, including the pre-FilteredIndex
        layout (whose live documents are re-indexed).

        Raises:
            ValueError: If no saved store is found, or the index and the
                documents disagree.
        """
        root = Path(path)
        store_file = root / _STORE_FILENAME
        if not store_file.exists():
            raise ValueError(f"No saved store found at {root}")
        data = json.loads(store_file.read_text(encoding="utf-8"))
        store = cls(**data["init_parameters"])
        documents = [Document.from_dict(d) for d in data["documents"]]

        if "format_version" not in data:
            # simlar <= 1.0: an append-only index plus tombstoned positions.
            deleted = set(data.get("deleted_positions", ()))
            live = [d for p, d in enumerate(documents) if p not in deleted]
            store.write_documents(live, policy=DuplicatePolicy.OVERWRITE)
            return store

        if documents:
            store._index = FilteredIndex.load(str(root / _INDEX_DIRNAME))
            if store._index.ids != [d.id for d in documents]:
                raise ValueError(f"Corrupted store at {root}: index and documents disagree.")
            store._dim = len(documents[0].embedding or ()) or None
        store._docs = documents
        store._pos = {d.id: p for p, d in enumerate(documents)}
        store._fields = MetaFields(
            kinds={k: set(v) for k, v in data["fields"]["kinds"].items()},
            columns=dict(data["fields"]["columns"]),
        )
        return store


def _hashable(value: Any) -> Any:
    """A hashable stand-in for a metadata value, keeping 1, 1.0, True and "1" apart."""
    try:
        hash(value)
        return (type(value).__name__, value)
    except TypeError:
        return (type(value).__name__, json.dumps(value, sort_keys=True, default=str))
