"""Haystack 3.x retriever components for SimlarDocumentStore.

``SimlarEmbeddingRetriever`` (vector), ``SimlarBM25Retriever`` (keyword) and
``SimlarHybridRetriever`` (BM25 + vector, RRF-fused). Each takes Haystack
``filters`` at init and/or run time, combined per ``filter_policy``, and
``roles`` for access control. Wire the hybrid one like this::

    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    pipeline.add_component("retriever", SimlarHybridRetriever(document_store=store))
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.run({
        "text_embedder": {"text": question},
        "retriever": {
            "query": question,
            "filters": {"field": "meta.lang", "operator": "==", "value": "en"},
            "roles": ["staff"],
        },
    })
"""

from __future__ import annotations

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.core.serialization import allow_deserialization_module
from haystack.document_stores.types import FilterPolicy, apply_filter_policy

from simlar.integrations.haystack.simlar_document_store import SimlarDocumentStore

# Let Pipeline.load()/loads() rebuild these components and the store, as it
# does for haystack_integrations.*. Scoped to this package only; their
# from_dict just calls __init__ with plain parameters.
allow_deserialization_module("simlar.integrations.haystack")


class _SimlarRetriever:
    """Init, serialization and argument resolution shared by the retrievers."""

    def __init__(
        self,
        document_store: SimlarDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: FilterPolicy | str = FilterPolicy.REPLACE,
        roles: list[str] | None = None,
        return_embedding: bool = False,
        parallel: bool | None = None,
    ):
        """
        Args:
            document_store: The SimlarDocumentStore to retrieve from.
            filters: Filters applied to every run (Haystack filter syntax).
            top_k: Default number of documents to return.
            filter_policy: How run-time ``filters`` combine with init ``filters``:
                ``REPLACE`` (default) or ``MERGE``.
            roles: Default roles for access control; ``None`` skips the check.
                A run-time ``roles`` replaces it.
            return_embedding: Return documents with their embedding.
            parallel: Threading mode for the search; ``None`` defers to the store.
        """
        if not isinstance(document_store, SimlarDocumentStore):
            raise TypeError("document_store must be an instance of SimlarDocumentStore")
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0. Currently, top_k is {top_k}")
        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.filter_policy = (
            FilterPolicy.from_str(filter_policy)
            if isinstance(filter_policy, str)
            else filter_policy
        )
        self.roles = roles
        self.return_embedding = return_embedding
        self.parallel = parallel

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            roles=self.roles,
            return_embedding=self.return_embedding,
            parallel=self.parallel,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return default_from_dict(cls, data)

    def _options(
        self,
        filters: dict[str, Any] | None,
        top_k: int | None,
        roles: list[str] | None,
        parallel: bool | None,
    ) -> dict[str, Any]:
        return {
            "filters": apply_filter_policy(self.filter_policy, self.filters, filters),
            "top_k": self.top_k if top_k is None else top_k,
            "roles": self.roles if roles is None else roles,
            "return_embedding": self.return_embedding,
            "parallel": self.parallel if parallel is None else parallel,
        }


@component
class SimlarEmbeddingRetriever(_SimlarRetriever):
    """Retrieves the documents most similar to a query embedding."""

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        roles: list[str] | None = None,
        parallel: bool | None = None,
    ) -> dict[str, list[Document]]:
        """
        Args:
            query_embedding: Embedding of the query, from an upstream text embedder.
            filters: Run-time filters, combined with init filters per ``filter_policy``.
            top_k: Override the init ``top_k``.
            roles: Override the init ``roles``.
            parallel: Override the init threading mode.
        """
        options = self._options(filters, top_k, roles, parallel)
        return {"documents": self.document_store.embedding_retrieval(query_embedding, **options)}


@component
class SimlarBM25Retriever(_SimlarRetriever):
    """Retrieves the documents that best match a query by BM25."""

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        roles: list[str] | None = None,
        parallel: bool | None = None,
    ) -> dict[str, list[Document]]:
        """
        Args:
            query: Query text.
            filters: Run-time filters, combined with init filters per ``filter_policy``.
            top_k: Override the init ``top_k``.
            roles: Override the init ``roles``.
            parallel: Override the init threading mode.
        """
        options = self._options(filters, top_k, roles, parallel)
        return {"documents": self.document_store.bm25_retrieval(query, **options)}


@component
class SimlarHybridRetriever(_SimlarRetriever):
    """Retrieves documents by BM25 and vector similarity, fused with RRF."""

    def __init__(
        self,
        document_store: SimlarDocumentStore,
        top_k: int = 5,
        parallel: bool | None = None,
        filters: dict[str, Any] | None = None,
        filter_policy: FilterPolicy | str = FilterPolicy.REPLACE,
        roles: list[str] | None = None,
        return_embedding: bool = False,
    ):
        """See :class:`SimlarEmbeddingRetriever`. ``top_k`` defaults to 5."""
        # Not super(): @component re-creates the class, which breaks its __class__ cell.
        _SimlarRetriever.__init__(
            self,
            document_store,
            filters=filters,
            top_k=top_k,
            filter_policy=filter_policy,
            roles=roles,
            return_embedding=return_embedding,
            parallel=parallel,
        )

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        parallel: bool | None = None,
        filters: dict[str, Any] | None = None,
        roles: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Args:
            query: Query text.
            query_embedding: Embedding of the query, from an upstream text embedder.
            top_k: Override the init ``top_k``.
            parallel: Override the init threading mode.
            filters: Run-time filters, combined with init filters per ``filter_policy``.
            roles: Override the init ``roles``.
        """
        options = self._options(filters, top_k, roles, parallel)
        return {
            "documents": self.document_store.hybrid_retrieval(query, query_embedding, **options)
        }
