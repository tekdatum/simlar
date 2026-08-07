"""
SimlarDocumentStore — a Haystack 2.x DocumentStore backed by simlar's StreamingHelixIndex.

Design principles:
- The store does NOT own an embedder. Documents must arrive with their embedding
  already set (via an upstream Haystack embedder component). This lets callers
  swap any embedder: SentenceTransformers, OpenAI, Cohere, etc.
- At search time, the caller provides both query text and a pre-computed query embedding.
- The underlying StreamingHelixIndex is append-only; deletions are handled via tombstones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError

from simlar.indexes.streaming_index import StreamingHelixIndex


class SimlarDocumentStore:
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
            relevance_k: Text candidate pool size fed into RRF.
            core_k: Vector candidate pool size fed into RRF.
            parallel: Default threading mode for writes and searches. Override
                per call with the ``parallel`` argument on :meth:`write_documents`
                and :meth:`search`.
        """
        self._top_k = top_k
        self._relevance_k = relevance_k
        self._core_k = core_k
        self._parallel = parallel
        self._index = StreamingHelixIndex(
            text_k=relevance_k,
            vector_k=core_k,
            top_k=top_k,
        )
        self._corpus: list[str] = []
        self._haystack_docs: list[Document] = []
        self._deleted_positions: set[int] = set()
        self._doc_id_to_pos: dict[str, int] = {}

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        parallel: bool | None = None,
    ) -> int:
        """Write documents to the store.

        Args:
            documents: Documents with their ``embedding`` field already set.
            policy: Haystack duplicate-handling policy.
            parallel: Thread the index write for this call. Defaults to the
                store-level setting.
        """
        to_write: list[Document] = []

        for doc in documents:
            if doc.id in self._doc_id_to_pos:
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"Document {doc.id!r} already exists")
                if policy == DuplicatePolicy.SKIP:
                    continue
                # OVERWRITE / NONE: tombstone old position so it is hidden from queries
                self._deleted_positions.add(self._doc_id_to_pos[doc.id])
            if doc.embedding is None:
                raise ValueError(
                    f"Document {doc.id!r} has no embedding. "
                    "Run a Haystack embedder component before writing to SimlarDocumentStore."
                )
            to_write.append(doc)

        if not to_write:
            return 0

        texts = [d.content or "" for d in to_write]
        vectors = np.array([d.embedding for d in to_write], dtype=np.float32)

        base_pos = len(self._corpus)
        self._index.add_batch(
            texts, vectors, self._parallel if parallel is None else parallel
        )
        self._corpus.extend(texts)
        self._haystack_docs.extend(to_write)
        for i, doc in enumerate(to_write):
            self._doc_id_to_pos[doc.id] = base_pos + i

        return len(to_write)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict | None = None,
        parallel: bool | None = None,
    ) -> list[Document]:
        """Hybrid search. Both query_text and query_embedding are required.

        Args:
            query_text: Raw query string
            query_embedding: Pre-computed query vector (must match indexed document dimension).
            top_k: Override the store-level top_k for this query.
            filters: Optional Haystack filter dict applied post-retrieval.
            parallel: Thread this search. Defaults to the store-level setting.

        Returns:
            List of Haystack Documents ranked by RRF-fused score, with original metadata preserved.
        """
        if not self._corpus:
            return []

        k = top_k or self._top_k
        # Fetch extra candidates to compensate for tombstoned / filtered-out results
        fetch_k = min(len(self._corpus), k * 10) if (self._deleted_positions or filters) else k
        query_vector = np.array(query_embedding, dtype=np.float32)

        ids, scores = self._index.search(
            query_text=query_text,
            query_vector=query_vector,
            k=fetch_k,
            parallel=self._parallel if parallel is None else parallel,
        )

        results: list[Document] = []
        for doc_id, score in zip(ids, scores, strict=False):
            pos = int(doc_id)
            if pos in self._deleted_positions:
                continue
            orig = self._haystack_docs[pos]
            if filters and not self._matches_filters(orig, filters):
                continue
            results.append(
                Document(
                    content=self._corpus[pos],
                    meta={
                        **orig.meta,
                        "rank": len(results) + 1,
                        "doc_id": pos,
                        "score": float(score),
                    },
                )
            )
            if len(results) >= k:
                break

        return results

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_documents(self, document_ids: list[str]) -> None:
        """Tombstone documents by ID. The append-only index is not modified."""
        for doc_id in document_ids:
            pos = self._doc_id_to_pos.pop(doc_id, None)
            if pos is not None:
                self._deleted_positions.add(pos)

    def delete_all_documents(self) -> None:
        """Reset the store and rebuild the index from scratch."""
        self._index = StreamingHelixIndex(
            text_k=self._relevance_k,
            vector_k=self._core_k,
            top_k=self._top_k,
        )
        self._corpus = []
        self._haystack_docs = []
        self._deleted_positions = set()
        self._doc_id_to_pos = {}

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """Delete all documents matching filters. Returns the number deleted."""
        docs = self.filter_documents(filters)
        self.delete_documents([doc.id for doc in docs])
        return len(docs)

    # ── Filter / Count ────────────────────────────────────────────────────────

    def filter_documents(self, filters: dict | None = None) -> list[Document]:
        active = [
            self._haystack_docs[pos]
            for pos in range(len(self._haystack_docs))
            if pos not in self._deleted_positions
        ]
        if not filters:
            return active
        return [doc for doc in active if self._matches_filters(doc, filters)]

    def count_documents(self) -> int:
        return len(self._haystack_docs) - len(self._deleted_positions)

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        return len(self.filter_documents(filters))

    # ── Update ────────────────────────────────────────────────────────────────

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """Update metadata in-place for all documents matching filters. Returns count updated."""
        docs = self.filter_documents(filters)
        for doc in docs:
            doc.meta.update(meta)
        return len(docs)

    # ── Metadata introspection ────────────────────────────────────────────────

    def get_metadata_fields_info(self) -> dict[str, dict[str, Any]]:
        """Infer and return the types of all metadata fields from active documents."""
        fields: dict[str, dict[str, Any]] = {}
        for doc in self.filter_documents():
            for key, value in doc.meta.items():
                if key not in fields:
                    type_name = type(value).__name__
                    if type_name == "str":
                        type_name = "keyword"
                    elif type_name == "int":
                        type_name = "long"
                    elif type_name == "bool":
                        type_name = "boolean"
                    fields[key] = {"type": type_name}
        return fields

    def get_metadata_field_min_max(self, field_name: str) -> dict[str, Any]:
        """Return the min and max for a metadata field across active documents."""
        values = [
            v
            for doc in self.filter_documents()
            if (v := self._get_doc_value(doc, field_name)) is not None
        ]
        if not values:
            return {"min": None, "max": None}
        return {"min": min(values), "max": max(values)}

    def get_metadata_field_unique_values(self, field_name: str) -> list[Any]:
        """Return all unique non-None values for a metadata field across active documents."""
        seen: set = set()
        for doc in self.filter_documents():
            v = self._get_doc_value(doc, field_name)
            if v is not None:
                seen.add(v)
        return list(seen)

    def count_unique_metadata_by_filter(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """Return count of unique values per field for documents matching filters."""
        docs = self.filter_documents(filters)
        return {
            field: len({v for doc in docs if (v := self._get_doc_value(doc, field)) is not None})
            for field in metadata_fields
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the index and document metadata to a directory.

        Args:
            path: Directory to write into. Created if it does not exist.
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        self._index.save(str(root / "index"))

        data = {
            "corpus": self._corpus,
            "documents": [doc.to_dict() for doc in self._haystack_docs],
            "deleted_positions": list(self._deleted_positions),
            "doc_id_to_pos": self._doc_id_to_pos,
            "init_parameters": {
                "top_k": self._top_k,
                "relevance_k": self._relevance_k,
                "core_k": self._core_k,
                "parallel": self._parallel,
            },
        }
        with open(root / "store.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> SimlarDocumentStore:
        """Load a previously saved store from a directory.

        Args:
            path: Directory previously written by :meth:`save`.

        Raises:
            ValueError: If the directory or required files are missing.
        """
        root = Path(path)
        store_file = root / "store.json"
        if not store_file.exists():
            raise ValueError(f"No saved store found at {root}")

        with open(store_file, encoding="utf-8") as f:
            data = json.load(f)

        params = data["init_parameters"]
        obj = cls(**params)
        obj._index = StreamingHelixIndex.load(str(root / "index"))
        obj._corpus = data["corpus"]
        obj._haystack_docs = [Document.from_dict(d) for d in data["documents"]]
        obj._deleted_positions = set(data["deleted_positions"])
        obj._doc_id_to_pos = data["doc_id_to_pos"]
        return obj

    # ── DocumentStore protocol ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "init_parameters": {
                "top_k": self._top_k,
                "relevance_k": self._relevance_k,
                "core_k": self._core_k,
                "parallel": self._parallel,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> SimlarDocumentStore:
        return cls(**data.get("init_parameters", {}))

    # ── Filter helpers (Haystack filter DSL) ──────────────────────────────────

    def _matches_filters(self, doc: Document, filters: dict[str, Any]) -> bool:
        return self._check_condition(doc, filters)

    @staticmethod
    def _get_doc_value(doc: Document, field: str) -> Any:
        if field == "content":
            return doc.content
        if field == "id":
            return doc.id
        if field.startswith("meta."):
            return doc.meta.get(field[5:])
        if hasattr(doc, field):
            return getattr(doc, field)
        return doc.meta.get(field)

    def _check_condition(self, doc: Document, condition: dict[str, Any]) -> bool:
        if "operator" not in condition and "conditions" not in condition:
            raise FilterError("Filter condition missing 'operator'")

        operator = condition.get("operator", "==")

        if operator == "AND":
            if "conditions" not in condition:
                raise FilterError("Missing 'conditions' for AND operator")
            return all(self._check_condition(doc, c) for c in condition["conditions"])
        if operator == "OR":
            if "conditions" not in condition:
                raise FilterError("Missing 'conditions' for OR operator")
            return any(self._check_condition(doc, c) for c in condition["conditions"])
        if operator == "NOT":
            if "conditions" not in condition:
                raise FilterError("Missing 'conditions' for NOT operator")
            conditions = condition["conditions"]
            if not isinstance(conditions, list) or not conditions:
                raise FilterError("NOT operator expects at least one condition")
            return not all(self._check_condition(doc, c) for c in conditions)

        # Leaf condition
        if "field" not in condition:
            raise FilterError("Missing 'field' in filter condition")
        field = condition["field"]
        if not isinstance(field, str):
            raise FilterError("'field' in filter condition must be a string")
        if "value" not in condition:
            raise FilterError("Missing 'value' in filter condition")
        value = condition["value"]

        doc_val = self._get_doc_value(doc, field)

        if operator in (">", ">=", "<", "<="):
            if doc_val is None or value is None:
                return False
            is_number = lambda v: isinstance(v, (int, float))  # noqa: E731
            if not (is_number(doc_val) and is_number(value)) and type(doc_val) is not type(value):
                raise FilterError(
                    f"Type mismatch: cannot compare {type(doc_val)} with {type(value)}"
                )
            try:
                if operator == ">":
                    return doc_val > value
                if operator == ">=":
                    return doc_val >= value
                if operator == "<":
                    return doc_val < value
                return doc_val <= value
            except TypeError as e:
                raise FilterError(f"Type mismatch in filter: {e}") from e

        if operator == "==":
            return doc_val == value
        if operator == "!=":
            return doc_val != value
        if operator == "in":
            if not isinstance(value, list):
                raise FilterError("Value for 'in' must be a list")
            return doc_val in value
        if operator == "not in":
            if not isinstance(value, list):
                raise FilterError("Value for 'not in' must be a list")
            return doc_val not in value

        return False
