"""Smoke tests for SimlarVectorStore."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from langchain_core.embeddings import Embeddings

from simlar.integrations.langchain.simlar_vector_store import SimlarVectorStore


class _ConstantEmbeddings(Embeddings):
    """Returns the same 8-dimensional unit vector for every input."""

    DIM = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [np.ones(self.DIM, dtype=np.float32).tolist() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return np.ones(self.DIM, dtype=np.float32).tolist()


@pytest.fixture()
def store():
    return SimlarVectorStore(embedding=_ConstantEmbeddings())


class TestSimlarVectorStore:
    def test_empty_search_returns_empty(self, store):
        assert store.similarity_search("query", k=5) == []

    def test_add_texts_returns_ids(self, store):
        ids = store.add_texts(["hello world", "foo bar"])
        assert len(ids) == 2

    def test_add_texts_with_explicit_ids(self, store):
        ids = store.add_texts(["cancer treatment", "ML transformers"], ids=["a", "b"])
        assert ids == ["a", "b"]

    def test_similarity_search_returns_documents(self, store):
        store.add_texts(["cancer treatment", "machine learning"], ids=["doc_0", "doc_1"])
        results = store.similarity_search("cancer", k=2)
        assert len(results) >= 1
        assert all(hasattr(d, "page_content") for d in results)

    def test_similarity_search_with_score(self, store):
        store.add_texts(["immunotherapy clinical trial"], ids=["doc_0"])
        results = store.similarity_search_with_score("immunotherapy", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert hasattr(doc, "page_content")
        assert isinstance(score, float)

    def test_save_and_load_roundtrip(self, store, tmp_path):
        store.add_texts(["hello simlar"], ids=["a"])
        store.save_local(str(tmp_path))
        loaded = SimlarVectorStore.load_local(str(tmp_path), embedding=_ConstantEmbeddings())
        assert len(loaded._ids) == 1
        assert loaded._ids[0] == "a"

    def test_from_texts_factory(self):
        s = SimlarVectorStore.from_texts(
            texts=["a", "b", "c"],
            embedding=_ConstantEmbeddings(),
        )
        assert len(s._ids) == 3

    def test_embeddings_property(self, store):
        assert store.embeddings is store._embedding

    def test_add_empty_list_returns_empty(self, store):
        ids = store.add_texts([])
        assert ids == []
