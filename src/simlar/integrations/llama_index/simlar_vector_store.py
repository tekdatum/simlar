"""
LlamaIndex integration for simlar — two entry points:

``SimlarVectorStore`` (BasePydanticVectorStore)
    Plugs into LlamaIndex's standard ``VectorStoreIndex`` pipeline.
    Handles ``add``, ``delete``, ``query``, ``persist``, and ``from_persist_*``.

Quick-start with the VectorStore::

    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from simlar.integrations.llama_index.simlar_retriever import SimlarVectorStore

    embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
    vector_store = SimlarVectorStore.from_texts(texts, ids, vectors)
    storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_ctx, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=5)

"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

try:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False
    BaseRetriever = object  # type: ignore[misc,assignment]
    BasePydanticVectorStore = object  # type: ignore[misc,assignment]

from simlar import HelixIndex

# ── SimlarVectorStore ──────────────────────────────────────────────────────────
# ── Internal helpers ───────────────────────────────────────────────────────────


def _assert_local_fs(fs: Any) -> None:
    """Raise NotImplementedError if ``fs`` is a non-local fsspec filesystem."""
    if fs is None:
        return
    try:
        from fsspec.implementations.local import LocalFileSystem

        if not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("SimlarVectorStore only supports local storage.")
    except ImportError:
        pass


class SimlarVectorStore(BasePydanticVectorStore):  # type: ignore[misc]
    """
    LlamaIndex ``BasePydanticVectorStore`` backed by simlar's HelixIndex.

    Args:
        index: A pre-built ``HelixIndex``.
        id_to_text: Mapping from document ID to raw text. Populated automatically
            by ``add()``; supply manually when constructing from an existing index.
    """

    stores_text: bool = True

    _index: Any = PrivateAttr()
    _id_to_text: dict[str, str] = PrivateAttr()

    def __init__(
        self,
        index: HelixIndex,
        id_to_text: dict[str, str] | None = None,
    ) -> None:
        if not _LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "llama-index-core is required. Install with: pip install llama-index-core"
            )
        super().__init__(stores_text=True)
        self._index = index
        self._id_to_text = id_to_text if id_to_text is not None else {}

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        ids: list[str],
        vectors: np.ndarray,
        relevance_k: int = 500,
        core_k: int = 200,
        top_k: int = 100,
    ) -> SimlarVectorStore:
        """Build a ``SimlarVectorStore`` from raw texts, IDs, and pre-computed vectors.

        Args:
            texts: Document texts
            ids: Unique string ID per document.
            vectors: Float32 array of shape ``(n_docs, dim)``.
            relevance_k: Text candidate pool size fed into RRF.
            core_k: Vector candidate pool size fed into RRF.
            top_k: Final result list length from the HelixIndex.
        """
        index = HelixIndex(text_k=relevance_k, vector_k=core_k, top_k=top_k)
        index.add(ids=ids, texts=texts, vectors=vectors)
        return cls(index=index, id_to_text=dict(zip(ids, texts, strict=False)))

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        fs: Any | None = None,
    ) -> SimlarVectorStore:
        """Load a previously saved store from a directory."""
        return cls.from_persist_path(persist_path=persist_dir, fs=fs)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Any | None = None,
    ) -> SimlarVectorStore:
        """Load a previously saved store from a directory written by ``persist()``.

        Args:
            persist_path: Directory previously written by :meth:`persist`.

        Raises:
            NotImplementedError: If a non-local ``fs`` is provided.
            ValueError: If no saved store exists at ``persist_path``.
        """
        _assert_local_fs(fs)
        path = Path(persist_path)
        if not (path / "index").exists():
            raise ValueError(f"No saved SimlarVectorStore found at {persist_path!r}.")
        index = HelixIndex.load(str(path / "index"))
        id_to_text_path = path / "id_to_text.json"
        id_to_text: dict[str, str] = {}
        if id_to_text_path.exists():
            with open(id_to_text_path, encoding="utf-8") as f:
                id_to_text = json.load(f)
        return cls(index=index, id_to_text=id_to_text)

    # ── Core interface ────────────────────────────────────────────────────────

    @property
    def client(self) -> HelixIndex:
        """Return the underlying ``HelixIndex``."""
        return self._index

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """Add nodes to the index.

        Each node must already have its ``embedding`` field set — run a LlamaIndex
        embedder (e.g. ``VectorStoreIndex`` with an ``embed_model``) before calling.

        Args:
            nodes: Nodes with embeddings set.

        Returns:
            List of node IDs that were written.
        """
        if not nodes:
            return []
        ids = [node.node_id for node in nodes]
        texts = [node.get_content() for node in nodes]
        embeddings = [node.get_embedding() for node in nodes]
        if any(e is None for e in embeddings):
            raise ValueError(
                "All nodes must have embeddings set before adding to SimlarVectorStore. "
                "Run an embedder component first."
            )
        vectors = np.array(embeddings, dtype=np.float32)
        self._index.add(ids=ids, texts=texts, vectors=vectors)
        for node_id, text in zip(ids, texts, strict=False):
            self._id_to_text[node_id] = text
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Not supported — HelixIndex is append-only."""
        raise NotImplementedError(f"Cannot delete {ref_doc_id!r}: HelixIndex is append-only.")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Hybrid search.

        Args:
            query: Must have ``query_embedding`` set. ``query_str`` is used for the
                text leg when provided; omitting it falls back to vector-only search.

        Returns:
            ``VectorStoreQueryResult`` with ``nodes``, ``similarities``, and ``ids``.
        """
        if query.query_embedding is None:
            raise ValueError("query.query_embedding is required for SimlarVectorStore.")
        query_vector = np.array(query.query_embedding, dtype=np.float32)
        results = self._index.search(
            query_text=query.query_str,
            query_vector=query_vector,
            k=query.similarity_top_k,
        )
        nodes = [
            TextNode(
                text=self._id_to_text.get(r.id, ""),
                id_=r.id,
                metadata={"rank": r.rank, "score": r.score},
            )
            for r in results
        ]
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=[r.score for r in results],
            ids=[r.id for r in results],
        )

    def persist(
        self,
        persist_path: str,
        fs: Any | None = None,
    ) -> None:
        """Save the index and text map to a directory.

        Args:
            persist_path: Directory to write into. Created if it does not exist.

        Raises:
            NotImplementedError: If a non-local ``fs`` is provided.
        """
        _assert_local_fs(fs)
        path = Path(persist_path)
        path.mkdir(parents=True, exist_ok=True)
        self._index.save(str(path / "index"))
        with open(path / "id_to_text.json", "w", encoding="utf-8") as f:
            json.dump(self._id_to_text, f)
