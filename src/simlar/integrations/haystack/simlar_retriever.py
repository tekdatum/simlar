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
        def __init__(self, document_store: SimlarDocumentStore, top_k: int = 5):
            self.document_store = document_store
            self.top_k = top_k

        @component.output_types(documents=list[Document])
        def run(
            self,
            query: str,
            query_embedding: list[float],
            top_k: int | None = None,
        ) -> dict:
            """
            Args:
                query: Raw query text
                query_embedding: Pre-computed embedding from an upstream embedder component.
                top_k: Override the retriever-level top_k for this call.
            """
            documents = self.document_store.search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k or self.top_k,
            )
            return {"documents": documents}

except ImportError:
    # haystack-ai not installed — define a plain class with the same interface
    # so the store and retriever can be used standalone without a Haystack Pipeline.
    class SimlarHybridRetriever:  # type: ignore[no-redef]
        def __init__(self, document_store: SimlarDocumentStore, top_k: int = 5):
            self.document_store = document_store
            self.top_k = top_k

        def run(
            self,
            query: str,
            query_embedding: list[float],
            top_k: int | None = None,
        ) -> dict:
            documents = self.document_store.search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k or self.top_k,
            )
            return {"documents": documents}
