"""Smoke tests for SimlarRetriever (LlamaIndex)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("llama_index.core", reason="llama-index-core not installed")

from llama_index.core.schema import NodeWithScore

from simlar.integrations.llama_index.simlar_retriever import SimlarRetriever

DIM = 8


class _ConstantEmbedding:
    """Returns the same 8-dimensional unit vector for every input."""

    def get_query_embedding(self, query: str) -> list[float]:
        return np.ones(DIM, dtype=np.float32).tolist()

    def get_text_embedding_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        return [np.ones(DIM, dtype=np.float32).tolist() for _ in texts]


_EMBED = _ConstantEmbedding()
_TEXTS = ["cancer treatment", "machine learning", "immunotherapy"]
_IDS = ["doc_0", "doc_1", "doc_2"]
_VECTORS = np.ones((len(_TEXTS), DIM), dtype=np.float32)


@pytest.fixture()
def retriever():
    return SimlarRetriever.from_texts(
        texts=_TEXTS,
        ids=_IDS,
        vectors=_VECTORS,
        embed_model=_EMBED,
        k=3,
    )


class TestSimlarRetriever:
    def test_from_texts_factory(self, retriever):
        assert retriever._k == 3
        assert len(retriever._id_to_text) == 3

    def test_retrieve_returns_nodes(self, retriever):
        nodes = retriever.retrieve("cancer")
        assert len(nodes) >= 1
        assert all(isinstance(n, NodeWithScore) for n in nodes)

    def test_retrieve_node_has_text(self, retriever):
        nodes = retriever.retrieve("cancer treatment")
        assert all(n.node.text for n in nodes)

    def test_retrieve_node_has_score(self, retriever):
        nodes = retriever.retrieve("query")
        assert all(isinstance(n.score, float) for n in nodes)

    def test_retrieve_respects_k(self):
        r = SimlarRetriever.from_texts(
            texts=_TEXTS, ids=_IDS, vectors=_VECTORS, embed_model=_EMBED, k=1
        )
        nodes = r.retrieve("cancer")
        assert len(nodes) <= 1

    def test_id_to_text_populated(self, retriever):
        for doc_id, text in zip(_IDS, _TEXTS, strict=False):
            assert retriever._id_to_text[doc_id] == text

    def test_embeddings_property(self, retriever):
        assert retriever._embed_model is _EMBED

    def test_direct_constructor(self):
        from simlar import HelixIndex

        index = HelixIndex(top_k=5)
        index.add(ids=_IDS, texts=_TEXTS, vectors=_VECTORS)
        r = SimlarRetriever(
            index=index,
            id_to_text=dict(zip(_IDS, _TEXTS, strict=False)),
            embed_model=_EMBED,
            k=2,
        )
        nodes = r.retrieve("immunotherapy")
        assert isinstance(nodes, list)


class TestSwappableIndexes:
    def test_default_text_index_is_relevance_core(self, retriever):
        # HelixIndex().text_index returns the compiled core object, not the
        # RelevanceIndex Python wrapper -- confirmed against the real engine.
        from simlar_engine.indexes._relevance_impl import _RelevanceCore

        assert isinstance(retriever._index.text_index, _RelevanceCore)

    def test_custom_text_index_via_retriever_from_texts(self):
        pytest.importorskip("bm25x", reason="bm25x not installed")
        from simlar.indexes.bm25x_index import BM25xIndex

        r = SimlarRetriever.from_texts(
            texts=_TEXTS, ids=_IDS, vectors=_VECTORS, embed_model=_EMBED, k=3,
            text_index=BM25xIndex(),
        )
        assert isinstance(r._index.text_index, BM25xIndex)

    def test_custom_text_index_via_vector_store_from_texts(self):
        pytest.importorskip("bm25x", reason="bm25x not installed")
        from simlar.indexes.bm25x_index import BM25xIndex
        from simlar.integrations.llama_index.simlar_vector_store import SimlarVectorStore

        store = SimlarVectorStore.from_texts(
            texts=_TEXTS, ids=_IDS, vectors=_VECTORS, text_index=BM25xIndex()
        )
        assert isinstance(store.client.text_index, BM25xIndex)


class TestRetrieverBatch:
    def test_retrieve_batch_matches_looped_retrieve(self, retriever):
        queries = ["cancer", "learning", "immunotherapy"]
        batched = retriever.retrieve_batch(queries)
        looped = [retriever.retrieve(q) for q in queries]
        assert [[n.node.node_id for n in nodes] for nodes in batched] == [
            [n.node.node_id for n in nodes] for nodes in looped
        ]

    def test_retrieve_batch_empty_returns_empty_list(self, retriever):
        assert retriever.retrieve_batch([]) == []


class TestVectorStoreQueryBatch:
    def _store(self):
        from llama_index.core.vector_stores.types import VectorStoreQuery

        from simlar.integrations.llama_index.simlar_vector_store import SimlarVectorStore

        store = SimlarVectorStore.from_texts(texts=_TEXTS, ids=_IDS, vectors=_VECTORS)
        return store, VectorStoreQuery

    def test_query_batch_matches_looped_query(self):
        store, VectorStoreQuery = self._store()
        embeddings = [_EMBED.get_query_embedding(q) for q in ["cancer", "learning"]]
        queries = [
            VectorStoreQuery(query_str="cancer", query_embedding=embeddings[0], similarity_top_k=2),
            VectorStoreQuery(query_str="learning", query_embedding=embeddings[1], similarity_top_k=2),
        ]
        batched = store.query_batch(queries)
        looped = [store.query(q) for q in queries]
        assert [r.ids for r in batched] == [r.ids for r in looped]

    def test_query_batch_requires_query_embedding(self):
        store, VectorStoreQuery = self._store()
        with pytest.raises(ValueError, match="query_embedding"):
            store.query_batch([VectorStoreQuery(query_str="cancer", similarity_top_k=2)])

    def test_query_batch_rejects_mixed_top_k(self):
        store, VectorStoreQuery = self._store()
        embedding = _EMBED.get_query_embedding("cancer")
        with pytest.raises(ValueError, match="similarity_top_k"):
            store.query_batch(
                [
                    VectorStoreQuery(query_str="a", query_embedding=embedding, similarity_top_k=1),
                    VectorStoreQuery(query_str="b", query_embedding=embedding, similarity_top_k=2),
                ]
            )

    def test_query_batch_empty_returns_empty_list(self):
        store, _VectorStoreQuery = self._store()
        assert store.query_batch([]) == []
