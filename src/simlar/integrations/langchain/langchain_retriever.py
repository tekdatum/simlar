"""SimlarRetriever — LangChain BaseRetriever backed by SimlarVectorStore.

Passes both the raw query string and its embedding
to the underlying HybridIndex, enabling RRF fusion across keyword and semantic signals.

You can also get a retriever directly from the store::

    retriever = store.as_retriever(search_kwargs={"k": 5})
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from simlar.integrations.langchain.simlar_vector_store import SimlarVectorStore


class SimlarRetriever(BaseRetriever):
    """BaseRetriever backed by a SimlarVectorStore.

    Example:
        .. code-block:: python

            from simlar.integrations.langchain.simlar_vector_store import SimlarVectorStore
            from simlar.integrations.langchain.langchain_retriever import SimlarRetriever
            from langchain_openai import OpenAIEmbeddings

            store = SimlarVectorStore.from_texts(texts, OpenAIEmbeddings())
            retriever = SimlarRetriever(vector_store=store, k=5)
            docs = retriever.invoke("cancer treatment")

    Attributes:
        parallel: Threading mode for the underlying search. ``None`` (default)
            defers to the store-level setting.
    """

    vector_store: SimlarVectorStore
    k: int = 5
    parallel: bool | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.vector_store.similarity_search(query, k=self.k, parallel=self.parallel)
