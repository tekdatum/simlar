#!/usr/bin/env python
# coding: utf-8

# # 10 — RAG Retriever Benchmark
#
# Converted from examples/10_rags_benchmark.ipynb (jupyter nbconvert --to script), with the
# %pip-install cell and the Jupyter-only `display()` calls adjusted to run as a plain script.
#
# This notebook evaluates simlar against other vector stores in a retrieval-augmented generation (RAG) setting. We measure how well each retriever finds the correct document for a given question — the core task in RAG before any LLM is involved.
#
# **What we cover**
# - Loading a question-answering dataset and building a ground-truth corpus
# - Evaluating retrieval quality with Hit@1, Hit@3, and MRR
# - Comparing simlar against Chroma, Qdrant, and Milvus
# - Measuring index build time and query latency

# In[1]:


# Install dependencies before running (was a `%pip install` cell in the notebook):
#   pip install -q \
#       sentence-transformers==5.6.0 datasets==5.0.0 \
#       langchain-core==1.4.8 langchain-huggingface==1.2.2 langchain-chroma==1.1.0 langchain-community==0.4.2 langchain-deepseek==1.1.0 \
#       qdrant-client==1.18.0 \
#       pymilvus==3.0.0 pymilvus[milvus_lite]==3.0.0 \
#       pinecone==9.1.0 \
#       llama-index-core==0.14.23 llama-index-embeddings-huggingface==0.7.0 llama-index-llms-openai-like==0.7.2 \
#       haystack-ai==2.31.0 numba==0.65.1 faiss-cpu==1.14.3 turbovec==0.8.0


# ## Environment variables
# 
# API keys for Pinecone and DeepSeek. Each prompt only appears if the key is not already set in the environment.

# In[1]:


import os
from getpass import getpass


if not os.environ.get("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass("Pinecone API key: ")
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass("DeepSeek API key: ")


# ## Dataset
# 
# We use [Kaggle](https://www.kaggle.com/datasets/ruhulaminsharif/squad-dataset) — a reading comprehension dataset where each question has a corresponding context passage that contains the answer. This structure maps naturally to RAG: the context passages become the corpus and the questions become the queries. For each query we know exactly which document should be retrieved, giving us a ground truth to evaluate against. Place the CSV file in the same directory as this notebook before running.

# In[1]:


from datasets import load_dataset
from abc import ABC, abstractmethod


class Dataset(ABC):
    corpus: list[str]
    queries: list[str]

    @abstractmethod
    def evaluate(self, query: str, docs: list[str]) -> bool: ...

    @abstractmethod
    def rank(self, query: str, docs: list[str]) -> int | None: ...


class SquadDataset(Dataset):
    def __init__(self, corpus, queries, query_to_context):
        self.corpus = corpus
        self.queries = queries
        self._query_to_context = query_to_context

    @classmethod
    def from_hf(cls, n_questions=None, n_contexts=None):
        hf = load_dataset("rajpurkar/squad", split="train")
        seen, corpus, queries, query_to_context = set(), [], [], {}
        for row in hf:
            if n_questions is not None and len(queries) == n_questions:
                break
            ctx = row["context"]
            if ctx not in seen:
                if n_contexts is not None and len(corpus) >= n_contexts:
                    continue
                seen.add(ctx)
                corpus.append(ctx)
            queries.append(row["question"])
            query_to_context[row["question"]] = ctx
        return cls(corpus=corpus, queries=queries, query_to_context=query_to_context)

    def evaluate(self, query, docs):
        return self._query_to_context.get(query) in docs

    def rank(self, query: str, docs: list[str]) -> int | None:
        correct = self._query_to_context.get(query)
        for i, doc in enumerate(docs):
            if doc == correct:
                return i
        return None


class MsMarcoDataset(Dataset):
    def __init__(self, corpus, queries, query_to_selected):
        self.corpus = corpus
        self.queries = queries
        self._query_to_selected = query_to_selected

    @classmethod
    def from_hf(cls, n_queries=None, n_passages=None):
        hf = load_dataset("microsoft/ms_marco", "v2.1", split="train")
        seen, passages, queries, query_to_selected = set(), [], [], {}

        for row in hf:
            if n_passages is None or len(passages) < n_passages:
                for text in row["passages"]["passage_text"]:
                    if text not in seen:
                        seen.add(text)
                        passages.append(text)

            if n_queries is None or len(queries) < n_queries:
                selected = {t for t, s in zip(row["passages"]["passage_text"], row["passages"]["is_selected"]) if s == 1}
                if selected and selected & set(passages):
                    queries.append(row["query"])
                    query_to_selected[row["query"]] = selected & set(passages)

            if (n_passages is None or len(passages) >= n_passages) and \
            (n_queries is None or len(queries) >= n_queries):
                break

        return cls(corpus=passages, queries=queries, query_to_selected=query_to_selected)

    def evaluate(self, query, docs):
        return bool(self._query_to_selected.get(query, set()) & set(docs))

    def rank(self, query, docs):
        selected = self._query_to_selected.get(query, set())
        for i, doc in enumerate(docs):
            if doc in selected:
                return i
        return None


# ## Suits
# 
# Each suit wraps a vector store and exposes a uniform interface: given a query string, return the top-k most relevant documents from the corpus. We test simlar across its three integration layers — LangChain, LlamaIndex, and Haystack — and compare against Chroma, Qdrant, and Milvus.
# 
# All suits use the same embedding model (`all-MiniLM-L6-v2`) so that differences in retrieval quality reflect the underlying index, not the embeddings.

# In[ ]:


from typing import Protocol

class HaystackBaseRetriever(Protocol):
    def run(self, query: str) -> dict: ...


# ### simlar
# 
# simlar exposes a hybrid index that combines keyword (BM25) and semantic (vector) search. It integrates natively with LangChain, LlamaIndex, and Haystack — the three classes below each wrap the same simlar index through a different integration layer.

# In[ ]:


from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangchainBaseRetriever
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle


class SimlarSuite:
    @classmethod
    def for_langchain(cls, dataset: Dataset, k: int = 3) -> LangchainBaseRetriever:
        from langchain_huggingface import HuggingFaceEmbeddings
        from simlar.integrations.langchain.simlar_vector_store import SimlarVectorStore
        from simlar.integrations.langchain.langchain_retriever import SimlarRetriever
        SimlarRetriever.model_rebuild()
        store = SimlarVectorStore.from_texts(
            texts=dataset.corpus,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )
        return SimlarRetriever(vector_store=store, k=k)

    @classmethod
    def for_llamaindex(cls, dataset: Dataset, k: int = 3) -> LlamaBaseRetriever:
        import numpy as np
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from simlar.integrations.llama_index.simlar_retriever import SimlarRetriever
        embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        vectors = np.array(embed_model.get_text_embedding_batch(dataset.corpus), dtype=np.float32)
        return SimlarRetriever.from_texts(
            texts=dataset.corpus,
            ids=[str(i) for i in range(len(dataset.corpus))],
            vectors=vectors,
            embed_model=embed_model,
            k=k,
        )

    @classmethod
    def for_haystack(cls, dataset: Dataset, k: int = 3) -> HaystackBaseRetriever:
        from haystack import Document as HaystackDocument, component
        from haystack.components.embedders import (
            SentenceTransformersDocumentEmbedder,
            SentenceTransformersTextEmbedder,
        )
        from simlar.integrations.haystack.simlar_document_store import SimlarDocumentStore
        from simlar.integrations.haystack.simlar_retriever import SimlarHybridRetriever

        store = SimlarDocumentStore(top_k=k)
        doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False)
        doc_embedder.warm_up()
        store.write_documents(
            doc_embedder.run([HaystackDocument(content=t) for t in dataset.corpus])["documents"]
        )
        _retriever = SimlarHybridRetriever(document_store=store, top_k=k)
        _text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False)
        _text_embedder.warm_up()

        @component
        class _R:
            @component.output_types(documents=list[HaystackDocument])
            def run(self, query: str) -> dict:
                embedding = _text_embedder.run(query)["embedding"]
                return _retriever.run(query=query, query_embedding=embedding)

        return _R()


# ### BM25x
#
# Same hybrid pipeline as simlar above (RRF-fused keyword + vector, `SimlarEngine` as the vector
# half, `all-MiniLM-L6-v2` embeddings) - only the keyword engine is swapped from the default
# `RelevanceIndex` to `BM25xIndex`, simlar's second `TextIndex` implementation, backed by the
# open-source `bm25x` library instead of the proprietary `simlar_engine`. Built directly against
# `HelixIndex`/`StreamingHybridIndex` (not through `SimlarVectorStore`/`SimlarDocumentStore`,
# which don't currently expose a way to override the keyword engine) - matching how every
# non-simlar suite below builds its own index/client directly rather than going through a shared
# store abstraction.

# In[ ]:


class BM25xSuite:
    @classmethod
    def for_langchain(cls, dataset: Dataset, k: int = 3) -> LangchainBaseRetriever:
        import numpy as np
        from langchain_huggingface import HuggingFaceEmbeddings
        from simlar import HelixIndex
        from simlar.indexes.bm25x_index import BM25xIndex
        _corpus = dataset.corpus
        _ids = [str(i) for i in range(len(_corpus))]
        _embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectors = np.array(_embedder.embed_documents(_corpus), dtype=np.float32)
        _index = HelixIndex(text_index=BM25xIndex())
        _index.add(_ids, _corpus, vectors)

        class _R(LangchainBaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager=None):
                qv = np.array(_embedder.embed_query(query), dtype=np.float32)
                results = _index.search(query_text=query, query_vector=qv, k=k)
                return [Document(page_content=_corpus[int(r.id)]) for r in results]

        return _R()

    @classmethod
    def for_llamaindex(cls, dataset: Dataset, k: int = 3) -> LlamaBaseRetriever:
        import numpy as np
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from simlar import HelixIndex
        from simlar.indexes.bm25x_index import BM25xIndex
        _corpus = dataset.corpus
        _ids = [str(i) for i in range(len(_corpus))]
        _embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        vectors = np.array(_embed_model.get_text_embedding_batch(_corpus), dtype=np.float32)
        _index = HelixIndex(text_index=BM25xIndex())
        _index.add(_ids, _corpus, vectors)

        class _R(LlamaBaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                qv = np.array(_embed_model.get_query_embedding(query_bundle.query_str), dtype=np.float32)
                results = _index.search(query_text=query_bundle.query_str, query_vector=qv, k=k)
                return [NodeWithScore(node=TextNode(text=_corpus[int(r.id)]), score=r.score) for r in results]

        return _R()

    @classmethod
    def for_haystack(cls, dataset: Dataset, k: int = 3) -> HaystackBaseRetriever:
        import numpy as np
        from haystack import Document as HaystackDocument, component
        from sentence_transformers import SentenceTransformer
        from simlar import StreamingHybridIndex
        from simlar.indexes.bm25x_index import BM25xIndex
        _corpus = dataset.corpus
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = _model.encode(_corpus, normalize_embeddings=True).astype(np.float32)
        _index = StreamingHybridIndex(text_index_cls=BM25xIndex)
        _index.add_batch(_corpus, vectors)

        @component
        class _R:
            @component.output_types(documents=list[HaystackDocument])
            def run(self, query: str) -> dict:
                qv = _model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
                ids, _scores = _index.search(query, qv, k=k)
                return {"documents": [HaystackDocument(content=_corpus[i]) for i in ids if i >= 0]}

        return _R()


# ### Chroma
# 
# [Chroma](https://www.trychroma.com/) is an open-source embedding database designed for AI applications. It stores vectors in memory or on disk and supports similarity search out of the box. In this benchmark we use it in in-memory mode so there is no persistence overhead between runs.

# In[ ]:


class ChromaSuite:
    @classmethod
    def for_langchain(cls, dataset: Dataset, k: int = 3) -> LangchainBaseRetriever:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        store = Chroma.from_texts(
            texts=dataset.corpus,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )
        return store.as_retriever(search_kwargs={"k": k})

    @classmethod
    def for_llamaindex(cls, dataset: Dataset, k: int = 3) -> LlamaBaseRetriever:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        _store = Chroma.from_texts(
            texts=dataset.corpus,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )

        class _R(LlamaBaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                return [NodeWithScore(node=TextNode(text=d.page_content))
                        for d in _store.similarity_search(query_bundle.query_str, k=k)]

        return _R()

    @classmethod
    def for_haystack(cls, dataset: Dataset, k: int = 3) -> HaystackBaseRetriever:
        from haystack import Document as HaystackDocument, component
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        _store = Chroma.from_texts(
            texts=dataset.corpus,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )

        @component
        class _R:
            @component.output_types(documents=list[HaystackDocument])
            def run(self, query: str) -> dict:
                return {"documents": [HaystackDocument(content=d.page_content)
                                      for d in _store.similarity_search(query, k=k)]}

        return _R()


# ### Qdrant
# 
# [Qdrant](https://qdrant.tech/) is a vector search engine built in Rust, optimized for high-performance similarity search. It supports filtering, payloads, and multiple distance metrics. Here we run it in in-memory mode (`":memory:"`) to keep setup simple and avoid disk I/O in the benchmark.

# In[ ]:


class QdrantSuite:
    @classmethod
    def for_langchain(cls, dataset: Dataset, k: int = 3) -> LangchainBaseRetriever:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        _client = QdrantClient(":memory:")
        _client.create_collection(
            collection_name="corpus",
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        _client.upsert(
            collection_name="corpus",
            points=[PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": text})
                    for i, text in enumerate(_corpus)],
        )

        class _R(LangchainBaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager=None):
                qv = _model.encode([query], normalize_embeddings=True)[0].tolist()
                return [Document(page_content=r.payload["text"])
                        for r in _client.query_points(collection_name="corpus", query=qv, limit=k).points]

        return _R()

    @classmethod
    def for_llamaindex(cls, dataset: Dataset, k: int = 3) -> LlamaBaseRetriever:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        _client = QdrantClient(":memory:")
        _client.create_collection(
            collection_name="corpus",
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        _client.upsert(
            collection_name="corpus",
            points=[PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": text})
                    for i, text in enumerate(_corpus)],
        )

        class _R(LlamaBaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                qv = _model.encode([query_bundle.query_str], normalize_embeddings=True)[0].tolist()
                results = _client.query_points(collection_name="corpus", query=qv, limit=k).points
                return [NodeWithScore(node=TextNode(text=r.payload["text"]), score=r.score) for r in results]

        return _R()

    @classmethod
    def for_haystack(cls, dataset: Dataset, k: int = 3) -> HaystackBaseRetriever:
        from haystack import Document as HaystackDocument, component
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        _client = QdrantClient(":memory:")
        _client.create_collection(
            collection_name="corpus",
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        _client.upsert(
            collection_name="corpus",
            points=[PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": text})
                    for i, text in enumerate(_corpus)],
        )

        @component
        class _R:
            @component.output_types(documents=list[HaystackDocument])
            def run(self, query: str) -> dict:
                qv = _model.encode([query], normalize_embeddings=True)[0].tolist()
                results = _client.query_points(collection_name="corpus", query=qv, limit=k).points
                return {"documents": [HaystackDocument(content=r.payload["text"]) for r in results]}

        return _R()


# ### Milvus
# 
# [Milvus](https://milvus.io/) is a distributed vector database built for large-scale similarity search. We use [Milvus Lite](https://milvus.io/docs/milvus_lite.md) — a lightweight version that runs locally as a file-based database, with no server required. Note that only one process can open the database file at a time.

# In[ ]:


class MilvusSuite:
    @classmethod
    def for_langchain(cls, dataset: Dataset, k: int = 3) -> LangchainBaseRetriever:
        from sentence_transformers import SentenceTransformer
        from pymilvus import MilvusClient
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        _client = MilvusClient("/tmp/milvus_rags.db")
        if _client.has_collection("corpus"):
            _client.drop_collection("corpus")
        _client.create_collection(collection_name="corpus", dimension=vectors.shape[1])
        _client.insert(
            collection_name="corpus",
            data=[{"id": i, "vector": vectors[i].tolist(), "text": text}
                  for i, text in enumerate(_corpus)],
        )

        class _R(LangchainBaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager=None):
                qv = _model.encode([query], normalize_embeddings=True)[0].tolist()
                results = _client.search(collection_name="corpus", data=[qv], limit=k, output_fields=["text"])
                return [Document(page_content=r["entity"]["text"]) for r in results[0]]

        return _R()

    @classmethod
    def for_llamaindex(cls, dataset: Dataset, k: int = 3) -> LlamaBaseRetriever:
        from sentence_transformers import SentenceTransformer
        from pymilvus import MilvusClient
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        _client = MilvusClient("/tmp/milvus_rags.db")
        if _client.has_collection("corpus"):
            _client.drop_collection("corpus")
        _client.create_collection(collection_name="corpus", dimension=vectors.shape[1])
        _client.insert(
            collection_name="corpus",
            data=[{"id": i, "vector": vectors[i].tolist(), "text": text}
                  for i, text in enumerate(_corpus)],
        )

        class _R(LlamaBaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                qv = _model.encode([query_bundle.query_str], normalize_embeddings=True)[0].tolist()
                results = _client.search(collection_name="corpus", data=[qv], limit=k, output_fields=["text"])
                return [NodeWithScore(node=TextNode(text=r["entity"]["text"])) for r in results[0]]

        return _R()

    @classmethod
    def for_haystack(cls, dataset: Dataset, k: int = 3) -> HaystackBaseRetriever:
        from haystack import Document as HaystackDocument, component
        from sentence_transformers import SentenceTransformer
        from pymilvus import MilvusClient
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        _client = MilvusClient("/tmp/milvus_rags.db")
        if _client.has_collection("corpus"):
            _client.drop_collection("corpus")
        _client.create_collection(collection_name="corpus", dimension=vectors.shape[1])
        _client.insert(
            collection_name="corpus",
            data=[{"id": i, "vector": vectors[i].tolist(), "text": text}
                  for i, text in enumerate(_corpus)],
        )

        @component
        class _R:
            @component.output_types(documents=list[HaystackDocument])
            def run(self, query: str) -> dict:
                qv = _model.encode([query], normalize_embeddings=True)[0].tolist()
                results = _client.search(collection_name="corpus", data=[qv], limit=k, output_fields=["text"])
                return {"documents": [HaystackDocument(content=r["entity"]["text"]) for r in results[0]]}

        return _R()


# ### Pinecone
# 
# [Pinecone](https://www.pinecone.io/) is a managed vector database — unlike the others, it runs as an external cloud service. This means build time includes network latency for uploading vectors, and query time includes a round-trip to Pinecone's servers. A `PINECONE_API_KEY` environment variable is required.

# In[ ]:


class PineconeSuite:
    @classmethod
    def for_langchain(cls, dataset: Dataset, k: int = 3) -> LangchainBaseRetriever:
        from sentence_transformers import SentenceTransformer
        from pinecone import Pinecone, ServerlessSpec
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        pc = Pinecone()
        if "rag-corpus" not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name="rag-corpus",
                dimension=vectors.shape[1],
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = pc.Index("rag-corpus")
        batch_size = 100
        records = [{"id": str(i), "values": vectors[i].tolist(), "metadata": {"text": text}}
                   for i, text in enumerate(_corpus)]
        for i in range(0, len(records), batch_size):
            _index.upsert(vectors=records[i:i + batch_size])

        class _R(LangchainBaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager=None):
                qv = _model.encode([query], normalize_embeddings=True)[0].tolist()
                results = _index.query(vector=qv, top_k=k, include_metadata=True)
                return [Document(page_content=m["metadata"]["text"]) for m in results["matches"]]

        return _R()

    @classmethod
    def for_llamaindex(cls, dataset: Dataset, k: int = 3) -> LlamaBaseRetriever:
        from sentence_transformers import SentenceTransformer
        from pinecone import Pinecone, ServerlessSpec
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        pc = Pinecone()
        if "rag-corpus" not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name="rag-corpus",
                dimension=vectors.shape[1],
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = pc.Index("rag-corpus")
        batch_size = 100
        records = [{"id": str(i), "values": vectors[i].tolist(), "metadata": {"text": text}}
                   for i, text in enumerate(_corpus)]
        for i in range(0, len(records), batch_size):
            _index.upsert(vectors=records[i:i + batch_size])

        class _R(LlamaBaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                qv = _model.encode([query_bundle.query_str], normalize_embeddings=True)[0].tolist()
                results = _index.query(vector=qv, top_k=k, include_metadata=True)
                return [NodeWithScore(node=TextNode(text=m["metadata"]["text"])) for m in results["matches"]]

        return _R()

    @classmethod
    def for_haystack(cls, dataset: Dataset, k: int = 3) -> HaystackBaseRetriever:
        from haystack import Document as HaystackDocument, component
        from sentence_transformers import SentenceTransformer
        from pinecone import Pinecone, ServerlessSpec
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True)
        pc = Pinecone()
        if "rag-corpus" not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name="rag-corpus",
                dimension=vectors.shape[1],
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = pc.Index("rag-corpus")
        batch_size = 100
        records = [{"id": str(i), "values": vectors[i].tolist(), "metadata": {"text": text}}
                   for i, text in enumerate(_corpus)]
        for i in range(0, len(records), batch_size):
            _index.upsert(vectors=records[i:i + batch_size])

        @component
        class _R:
            @component.output_types(documents=list[HaystackDocument])
            def run(self, query: str) -> dict:
                qv = _model.encode([query], normalize_embeddings=True)[0].tolist()
                results = _index.query(vector=qv, top_k=k, include_metadata=True)
                return {"documents": [HaystackDocument(content=m["metadata"]["text"]) for m in results["matches"]]}

        return _R()


# ### FAISS
# 
# [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. It runs entirely in memory with no server required, using optimized CPU/GPU kernels. Unlike the other stores in this benchmark, FAISS is pure vector search — no keyword component — making it a useful baseline for semantic-only retrieval.

# In[ ]:


class FaissSuite:
    @classmethod
    def for_langchain(cls, dataset: Dataset, k: int = 3) -> LangchainBaseRetriever:
        from langchain_community.vectorstores import FAISS as FaissStore
        from langchain_huggingface import HuggingFaceEmbeddings
        store = FaissStore.from_texts(
            texts=dataset.corpus,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )
        return store.as_retriever(search_kwargs={"k": k})

    @classmethod
    def for_llamaindex(cls, dataset: Dataset, k: int = 3) -> LlamaBaseRetriever:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True).astype(np.float32)
        _index = faiss.IndexFlatIP(vectors.shape[1])
        _index.add(vectors)

        class _R(LlamaBaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                qv = _model.encode([query_bundle.query_str], normalize_embeddings=True).astype(np.float32)
                scores, indices = _index.search(qv, k)
                return [NodeWithScore(node=TextNode(text=_corpus[i]), score=float(scores[0][j]))
                        for j, i in enumerate(indices[0]) if i >= 0]
        return _R()

    @classmethod
    def for_haystack(cls, dataset: Dataset, k: int = 3) -> HaystackBaseRetriever:
        import faiss
        import numpy as np
        from haystack import Document as HaystackDocument, component
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True).astype(np.float32)
        _index = faiss.IndexFlatIP(vectors.shape[1])
        _index.add(vectors)

        @component
        class _R:
            @component.output_types(documents=list[HaystackDocument])
            def run(self, query: str) -> dict:
                qv = _model.encode([query], normalize_embeddings=True).astype(np.float32)
                _, indices = _index.search(qv, k)
                return {"documents": [HaystackDocument(content=_corpus[i]) for i in indices[0] if i >= 0]}
        return _R()


# ### turbovec
# 
# [turbovec](https://github.com/RyanCodrai/turbovec) is a vector search library implemented in Rust with Python bindings, built on Google Research's TurboQuant algorithm. It uses 2–4 bit quantization to achieve up to 16x compression — a 10M document corpus that takes 31GB as float32 fits in 4GB. Hand-optimized SIMD kernels (AVX-512, NEON) make it faster than FAISS on ARM. Like FAISS, it is pure vector search with no keyword component.

# In[ ]:


class TurbovecSuite:
    @classmethod
    def for_langchain(cls, dataset: Dataset, k: int = 3) -> LangchainBaseRetriever:
        from turbovec.langchain import TurboQuantVectorStore
        from langchain_huggingface import HuggingFaceEmbeddings
        store = TurboQuantVectorStore.from_texts(
            texts=dataset.corpus,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            bit_width=4,
        )
        return store.as_retriever(search_kwargs={"k": k})

    @classmethod
    def for_llamaindex(cls, dataset: Dataset, k: int = 3) -> LlamaBaseRetriever:
        import numpy as np
        from turbovec import TurboQuantIndex
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True).astype(np.float32)
        _index = TurboQuantIndex(dim=vectors.shape[1], bit_width=4)
        _index.add(vectors)

        class _R(LlamaBaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                qv = _model.encode([query_bundle.query_str], normalize_embeddings=True).astype(np.float32)
                scores, indices = _index.search(qv, k=k)
                return [NodeWithScore(node=TextNode(text=_corpus[i]), score=float(scores[0][j]))
                        for j, i in enumerate(indices[0]) if i >= 0]
        return _R()

    @classmethod
    def for_haystack(cls, dataset: Dataset, k: int = 3) -> HaystackBaseRetriever:
        import numpy as np
        from turbovec import TurboQuantIndex
        from haystack import Document as HaystackDocument, component
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _corpus = dataset.corpus
        vectors = _model.encode(_corpus, normalize_embeddings=True).astype(np.float32)
        _index = TurboQuantIndex(dim=vectors.shape[1], bit_width=4)
        _index.add(vectors)

        @component
        class _R:
            @component.output_types(documents=list[HaystackDocument])
            def run(self, query: str) -> dict:
                qv = _model.encode([query], normalize_embeddings=True).astype(np.float32)
                _, indices = _index.search(qv, k=k)
                return {"documents": [HaystackDocument(content=_corpus[i]) for i in indices[0] if i >= 0]}
        return _R()


# ## RAG
# 
# Each RAG class wraps a retriever and exposes two methods:
# - `retrieve(query)` — returns the top-k documents as strings, no LLM involved. Used for the benchmark.
# - `ask(query)` — runs the full RAG pipeline with an LLM and returns an answer.

# In[ ]:


from dataclasses import dataclass


@dataclass
class AskResult:
    question: str
    answer: str | None
    docs: list[str]
    hit: bool


class RAG:
    def retrieve(self, query: str) -> list[str]: ...
    def ask(self, query: str) -> AskResult: ...


# ### LangChain RAG
# 
# Wraps a LangChain retriever. `retrieve()` calls the retriever directly. `ask()` runs the full chain with an LLM.

# In[ ]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_deepseek import ChatDeepSeek


class LangchainRAG(RAG):
    def __init__(self, dataset: Dataset, retriever: LangchainBaseRetriever):
        self._dataset = dataset
        self._retriever = retriever
        prompt = ChatPromptTemplate.from_template(
            "Answer using only the context below. Be concise.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )
        self._chain = (
            RunnableParallel(docs=retriever, question=RunnablePassthrough())
            | RunnablePassthrough.assign(context=lambda x: "\n\n".join(d.page_content for d in x["docs"]))
            | RunnableParallel(
                answer=prompt | ChatDeepSeek(model="deepseek-chat") | StrOutputParser(),
                docs=lambda x: x["docs"],
            )
        )

    def retrieve(self, query: str) -> list[str]:
        return [d.page_content for d in self._retriever.invoke(query)]

    def ask(self, query: str) -> AskResult:
        result = self._chain.invoke(query)
        docs = [d.page_content for d in result["docs"]]
        return AskResult(question=query, answer=result["answer"], docs=docs,
                         hit=self._dataset.evaluate(query, docs))


# ### LlamaIndex RAG
# 
# Wraps a LlamaIndex retriever. `retrieve()` calls the retriever directly. `ask()` runs the full query engine with an LLM.

# In[ ]:


from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever


class LlamaIndexRAG(RAG):
    def __init__(self, dataset: Dataset, retriever: LlamaBaseRetriever):
        self._dataset = dataset
        self._retriever = retriever
        from llama_index.core import Settings
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.openai_like import OpenAILike
        Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        Settings.llm = OpenAILike(
            model="deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key=__import__("os").environ["DEEPSEEK_API_KEY"],
            context_window=64000,
            is_chat_model=True,
        )
        self._query_engine = RetrieverQueryEngine.from_args(retriever=retriever)

    def retrieve(self, query: str) -> list[str]:
        return [n.node.text for n in self._retriever.retrieve(query)]

    def ask(self, query: str) -> AskResult:
        response = self._query_engine.query(query)
        docs = [n.node.text for n in response.source_nodes]
        return AskResult(question=query, answer=str(response), docs=docs,
                         hit=self._dataset.evaluate(query, docs))


# ### Haystack RAG
# 
# Wraps a Haystack retriever. `retrieve()` calls the retriever directly. `ask()` runs a Haystack pipeline with an LLM.

# In[ ]:


class HaystackRAG(RAG):
    def __init__(self, dataset: Dataset, retriever):
        self._dataset = dataset
        self._retriever = retriever
        from haystack import Pipeline
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.utils import Secret
        self._pipeline = Pipeline()
        self._pipeline.add_component("prompt", PromptBuilder(
            template="Answer using only the context below. Be concise.\n\nContext:\n{% for d in documents %}{{ d.content }}\n{% endfor %}\nQuestion: {{ query }}",
            required_variables=["documents", "query"],
        ))
        self._pipeline.add_component("generator", OpenAIChatGenerator(
            model="deepseek-chat",
            api_base_url="https://api.deepseek.com",
            api_key=Secret.from_env_var("DEEPSEEK_API_KEY"),
        ))
        self._pipeline.connect("prompt.prompt", "generator.messages")

    def retrieve(self, query: str) -> list[str]:
        return [d.content for d in self._retriever.run(query)["documents"]]

    def ask(self, query: str) -> AskResult:
        docs = self._retriever.run(query)["documents"]
        result = self._pipeline.run({"prompt": {"documents": docs, "query": query}})
        answer = result["generator"]["replies"][0].text
        doc_texts = [d.content for d in docs]
        return AskResult(question=query, answer=answer, docs=doc_texts,
                         hit=self._dataset.evaluate(query, doc_texts))


# ## Benchmark
# 
# `test_retrievers` runs the full benchmark in two phases:
# 
# 1. **Build** — each retriever indexes the corpus and the elapsed time is recorded.
# 2. **Query** — every question in the dataset is passed to `rag.retrieve()` and the result is evaluated against the ground truth.
# 
# ### Metrics
# 
# - **Hit@1** — fraction of queries where the correct document is ranked first.
# - **Hit@k** — fraction of queries where the correct document appears in the top k.
# - **MRR** (Mean Reciprocal Rank) — average of `1/rank` for the correct document. Penalizes results that are correct but ranked lower.

# In[ ]:


import io
import time
import contextlib
import pandas as pd
from typing import Callable

@contextlib.contextmanager
def _suppress():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)

def test_retrievers(dataset: SquadDataset, out:str, k: int = 3) -> None:
    print(f"Dataset: {len(dataset.corpus)} documents & {len(dataset.queries)} queries & {k} k")

    entries: list[tuple[str, Callable[[], object], Callable[[object], RAG]]] = [
        ("simlar-langchain", lambda: SimlarSuite.for_langchain(dataset, k), lambda r: LangchainRAG(dataset, r)),
        ("simlar-llamaindex", lambda: SimlarSuite.for_llamaindex(dataset, k), lambda r: LlamaIndexRAG(dataset, r)),
        ("simlar-haystack", lambda: SimlarSuite.for_haystack(dataset, k), lambda r: HaystackRAG(dataset, r)),
        ("bm25x-langchain", lambda: BM25xSuite.for_langchain(dataset, k), lambda r: LangchainRAG(dataset, r)),
        ("bm25x-llamaindex", lambda: BM25xSuite.for_llamaindex(dataset, k), lambda r: LlamaIndexRAG(dataset, r)),
        ("bm25x-haystack", lambda: BM25xSuite.for_haystack(dataset, k), lambda r: HaystackRAG(dataset, r)),
        ("chroma-langchain", lambda: ChromaSuite.for_langchain(dataset, k), lambda r: LangchainRAG(dataset, r)),
        ("chroma-llamaindex", lambda: ChromaSuite.for_llamaindex(dataset, k), lambda r: LlamaIndexRAG(dataset, r)),
        ("chroma-haystack", lambda: ChromaSuite.for_haystack(dataset, k), lambda r: HaystackRAG(dataset, r)),
        ("qdrant-langchain", lambda: QdrantSuite.for_langchain(dataset, k), lambda r: LangchainRAG(dataset, r)),
        ("qdrant-llamaindex", lambda: QdrantSuite.for_llamaindex(dataset, k), lambda r: LlamaIndexRAG(dataset, r)),
        ("qdrant-haystack", lambda: QdrantSuite.for_haystack(dataset, k), lambda r: HaystackRAG(dataset, r)),
        ("milvus-langchain", lambda: MilvusSuite.for_langchain(dataset, k), lambda r: LangchainRAG(dataset, r)),
        ("milvus-llamaindex", lambda: MilvusSuite.for_llamaindex(dataset, k), lambda r: LlamaIndexRAG(dataset, r)),
        ("milvus-haystack", lambda: MilvusSuite.for_haystack(dataset, k), lambda r: HaystackRAG(dataset, r)),
        # ("pinecone-langchain", lambda: PineconeSuite.for_langchain(dataset, k), lambda r: LangchainRAG(dataset, r)),
        # ("pinecone-llamaindex", lambda: PineconeSuite.for_llamaindex(dataset, k), lambda r: LlamaIndexRAG(dataset, r)),
        # ("pinecone-haystack", lambda: PineconeSuite.for_haystack(dataset, k), lambda r: HaystackRAG(dataset, r)),
        ("faiss-langchain", lambda: FaissSuite.for_langchain(dataset, k), lambda r: LangchainRAG(dataset, r)),
        ("faiss-llamaindex", lambda: FaissSuite.for_llamaindex(dataset, k), lambda r: LlamaIndexRAG(dataset, r)),
        ("faiss-haystack", lambda: FaissSuite.for_haystack(dataset, k), lambda r: HaystackRAG(dataset, r)),
        ("turbovec-langchain", lambda: TurbovecSuite.for_langchain(dataset, k), lambda r: LangchainRAG(dataset, r)),
        ("turbovec-llamaindex", lambda: TurbovecSuite.for_llamaindex(dataset, k), lambda r: LlamaIndexRAG(dataset, r)),
        ("turbovec-haystack", lambda: TurbovecSuite.for_haystack(dataset, k), lambda r: HaystackRAG(dataset, r)),
    ]

    # Build phase
    rags: list[dict] = []
    for name, retriever_factory, rag_factory in entries:
        print(f"Building {name}...")
        try:
            with _suppress():
                t0 = time.perf_counter()
                retriever = retriever_factory()
                build_ms = (time.perf_counter() - t0) * 1000
                rag = rag_factory(retriever)
            rags.append({"name": name, "rag": rag, "build_ms": build_ms})
        except Exception as e:
            print(f"  skipped: {e}")

    # Query phase
    n = len(dataset.queries)
    for entry in rags:
        name, rag = entry["name"], entry["rag"]
        hit1 = hitk = mrr = 0.0
        errors = 0
        t0 = time.perf_counter()
        for i, q in enumerate(dataset.queries, 1):
            try:
                with _suppress():
                    docs = rag.retrieve(q)
                    r = dataset.rank(q, docs)
                if r is not None:
                    if r == 0: hit1 += 1
                    hitk += 1
                    mrr += 1 / (r + 1)
            except Exception as e:
                errors += 1
                with open("errors.txt", "a") as f:
                    f.write(f"[{name}] query={repr(q)} error={e}\n")
            if i % 100 == 0 or i == n:
                print(f"\r  {name}: {i}/{n} queries{f' ({errors} errors)' if errors else ''}", end="", flush=True)
        print()
        entry["query_ms"] = (time.perf_counter() - t0) * 1000 / n
        entry["hit1"] = hit1 / n
        entry["hitk"] = hitk / n
        entry["mrr"]  = mrr  / n

    # Results
    hitk_col = f"Hit@{k}"
    rows = [
        {
            "Retriever": e["name"],
            "Hit@1":     e["hit1"],
            hitk_col:    e["hitk"],
            "MRR":       e["mrr"],
            "Build(ms)": e["build_ms"],
            "Query(ms)": e["query_ms"],
        }
        for e in rags
    ]
    df = pd.DataFrame(rows).set_index("Retriever")

    for suffix, label in [("langchain", "LangChain"), ("llamaindex", "LlamaIndex"), ("haystack", "Haystack")]:
        sub = df[df.index.str.endswith(f"-{suffix}")].copy()
        sub.index = sub.index.str.replace(f"-{suffix}", "", regex=False)
        print(f"\n### {label}")
        formatted = sub.copy()
        formatted["Hit@1"] = formatted["Hit@1"].map("{:.1%}".format)
        formatted[hitk_col] = formatted[hitk_col].map("{:.1%}".format)
        formatted["MRR"] = formatted["MRR"].map("{:.3f}".format)
        formatted["Build(ms)"] = formatted["Build(ms)"].map("{:.0f}".format)
        formatted["Query(ms)"] = formatted["Query(ms)"].map("{:.1f}".format)
        print(formatted.to_string())

    # Save to CSV
    df.reset_index().rename(columns={
        "Retriever": "name", "Hit@1": "hit1", hitk_col: "hitk",
        "MRR": "mrr", "Build(ms)": "build_ms", "Query(ms)": "query_ms",
    }).to_csv(out, index=False)
    print(f"\nSaved to {out}")


# In[ ]:


# dataset = SquadDataset.from_hf(n_questions=100)
dataset = MsMarcoDataset.from_hf(n_passages=500_000, n_queries=20_000)
test_retrievers(dataset, "benchmark_results.csv", k=10)

