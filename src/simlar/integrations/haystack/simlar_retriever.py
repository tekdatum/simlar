"""
SimlarHybridRetriever — a Haystack 2.x @component
In a Haystack Pipeline, wire it like this:

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.run({
        "text_embedder": {"text": question},
        "retriever": {"query": question},   # query text
    })
"""

from __future__ import annotations

from simlar.integrations.haystack.simlar_document_store import Document, SimlarDocumentStore

try:
    from haystack import component

    @component
    class SimlarHybridRetriever:
        def __init__(
            self,
            document_store: SimlarDocumentStore,
            top_k: int = 5,
            parallel: bool | None = None,
        ):
            """
            Args:
                document_store: Store to retrieve from.
                top_k: Default number of documents to return.
                parallel: Threading mode for the search. ``None`` defers to the
                    store-level setting.
            """
            self.document_store = document_store
            self.top_k = top_k
            self.parallel = parallel

        @component.output_types(documents=list[Document])
        def run(
            self,
            query: str,
            query_embedding: list[float],
            top_k: int | None = None,
            parallel: bool | None = None,
        ) -> dict:
            """
            Args:
                query: Raw query text
                query_embedding: Pre-computed embedding from an upstream embedder component.
                top_k: Override the retriever-level top_k for this call.
                parallel: Override the retriever-level threading mode for this call.
            """
            documents = self.document_store.search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k or self.top_k,
                parallel=self.parallel if parallel is None else parallel,
            )
            return {"documents": documents}

        def run_batch(
            self,
            queries: list[str],
            query_embeddings: list[list[float]],
            top_k: int | None = None,
        ) -> dict:
            """Batched sibling of run(): one StreamingHelixIndex call for every
            query, not one call per query.

            Args:
                queries: Raw query strings.
                query_embeddings: Pre-computed embeddings, one per query, same
                    order as queries, from an upstream embedder component.
                top_k: Override the retriever-level top_k for this call.
            """
            documents = self.document_store.search_batch(
                query_texts=queries,
                query_embeddings=query_embeddings,
                top_k=top_k or self.top_k,
            )
            return {"documents": documents}

except ImportError:
    # haystack-ai not installed — define a plain class with the same interface
    # so the store and retriever can be used standalone without a Haystack Pipeline.
    class SimlarHybridRetriever:  # type: ignore[no-redef]
        def __init__(
            self,
            document_store: SimlarDocumentStore,
            top_k: int = 5,
            parallel: bool | None = None,
        ):
            self.document_store = document_store
            self.top_k = top_k
            self.parallel = parallel

        def run(
            self,
            query: str,
            query_embedding: list[float],
            top_k: int | None = None,
            parallel: bool | None = None,
        ) -> dict:
            documents = self.document_store.search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k or self.top_k,
                parallel=self.parallel if parallel is None else parallel,
            )
            return {"documents": documents}

        def run_batch(
            self,
            queries: list[str],
            query_embeddings: list[list[float]],
            top_k: int | None = None,
        ) -> dict:
            documents = self.document_store.search_batch(
                query_texts=queries,
                query_embeddings=query_embeddings,
                top_k=top_k or self.top_k,
            )
            return {"documents": documents}
