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
