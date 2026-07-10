"""
``SimlarRetriever`` (BaseRetriever)
    Standalone retriever for direct use without a ``VectorStoreIndex``.
    Build the HelixIndex externally, or use ``SimlarRetriever.from_texts()``.

Quick-start with the standalone Retriever::

    from simlar.integrations.llama_index.simlar_retriever import SimlarRetriever

    retriever = SimlarRetriever.from_texts(
        texts=texts, ids=ids, vectors=vectors, embed_model=embed_model, k=5
    )
    nodes = retriever.retrieve("your question here")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.schema import QueryBundle

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False
    BaseRetriever = object  # type: ignore[misc,assignment]
    BasePydanticVectorStore = object  # type: ignore[misc,assignment]

from simlar import HelixIndex


class SimlarRetriever(BaseRetriever):  # type: ignore[misc]
    """LlamaIndex standalone retriever backed by simlar's HelixIndex."""

    def __init__(
        self,
        index: HelixIndex,
        id_to_text: dict[str, str],
        embed_model: BaseEmbedding,
        k: int = 5,
    ) -> None:
        """
        Args:
            index: A pre-built HelixIndex (see ``HelixIndex.add()``).
            id_to_text: Mapping from document ID to document text, used to
                populate the ``TextNode.text`` field in returned nodes.
            embed_model: Any LlamaIndex ``BaseEmbedding`` instance — used to
                embed the query string at retrieval time. Must have the same
                output dimension as the vectors stored in the index.
            k: Number of results to return.
        """
        if not _LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "llama-index-core is required. Install with: pip install llama-index-core"
            )
        self._index = index
        self._id_to_text = id_to_text
        self._embed_model = embed_model
        self._k = k
        super().__init__()

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        ids: list[str],
        vectors: np.ndarray,
        embed_model: BaseEmbedding,
        k: int = 5,
        relevance_k: int = 500,
        core_k: int = 200,
        top_k: int = 100,
    ) -> SimlarRetriever:
        """Build a ``SimlarRetriever`` from raw texts, IDs, and pre-computed vectors.

        Args:
            texts: Document texts (corpus).
            ids: Unique string ID per document.
            vectors: Float32 array of shape ``(n_docs, dim)`` — must match the
                output dimension of ``embed_model``.
            embed_model: LlamaIndex embedding model used to embed queries.
            k: Number of results to return.
            relevance_k: Text candidate pool size fed into RRF.
            core_k: Vector candidate pool size fed into RRF.
            top_k: Final result list length from the HelixIndex.
        """
        index = HelixIndex(text_k=relevance_k, vector_k=core_k, top_k=top_k)
        index.add(ids=ids, texts=texts, vectors=vectors)
        return cls(
            index=index,
            id_to_text=dict(zip(ids, texts, strict=False)),
            embed_model=embed_model,
            k=k,
        )

    @classmethod
    def from_persist(
        cls,
        directory: str,
        embed_model: BaseEmbedding,
        k: int = 5,
    ) -> SimlarRetriever:
        """Load a previously saved retriever from disk.

        Args:
            directory: Directory previously written by :meth:`persist`.
            embed_model: Embedding model used to embed queries at retrieval time.
                Must match the model used when the index was originally built.
            k: Number of results to return.

        Raises:
            ValueError: If no saved retriever exists at ``directory``.
        """
        path = Path(directory)
        if not (path / "index").exists():
            raise ValueError(f"No saved SimlarRetriever found at {directory!r}.")
        index = HelixIndex.load(str(path / "index"))
        id_to_text_path = path / "id_to_text.json"
        id_to_text: dict[str, str] = {}
        if id_to_text_path.exists():
            with open(id_to_text_path, encoding="utf-8") as f:
                id_to_text = json.load(f)
        return cls(index=index, id_to_text=id_to_text, embed_model=embed_model, k=k)

    # ── Persistence ───────────────────────────────────────────────────────────

    @property
    def client(self) -> HelixIndex:
        """Return the underlying ``HelixIndex``."""
        return self._index

    def persist(self, directory: str) -> None:
        """Save the index and text map to a directory.

        Args:
            directory: Directory to write into. Created if it does not exist.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._index.save(str(path / "index"))
        with open(path / "id_to_text.json", "w", encoding="utf-8") as f:
            json.dump(self._id_to_text, f)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_nodes(self, results) -> list[NodeWithScore]:
        return [
            NodeWithScore(
                node=TextNode(
                    text=self._id_to_text.get(r.id, ""),
                    id_=r.id,
                    metadata={"rank": r.rank, "score": r.score},
                ),
                score=r.score,
            )
            for r in results
        ]

    # ── BaseRetriever protocol ────────────────────────────────────────────────

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_vector = np.array(
            self._embed_model.get_query_embedding(query_bundle.query_str),
            dtype=np.float32,
        )
        results = self._index.search(
            query_text=query_bundle.query_str,
            query_vector=query_vector,
            k=self._k,
        )
        return self._to_nodes(results)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # get_query_embedding is synchronous in all current LlamaIndex embedders
        return self._retrieve(query_bundle)
