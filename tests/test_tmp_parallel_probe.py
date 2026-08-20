import numpy as np
import pytest

SEEN = []


@pytest.fixture(autouse=True)
def spy():
    import simlar_engine.indexes._helix_impl as H
    import simlar_engine.indexes._streaming_impl as S

    originals = []
    for cls, meth in (
        (H._HelixCore, "add"),
        (H._HelixCore, "search"),
        (S._StreamingCore, "add_batch"),
        (S._StreamingCore, "search"),
    ):
        orig = getattr(cls, meth)
        originals.append((cls, meth, orig))
        label = f"{cls.__name__}.{meth}"

        def make(orig=orig, label=label):
            def f(self, *a, **kw):
                p = kw.get("parallel", a[-1] if a and isinstance(a[-1], bool) else "MISSING")
                SEEN.append((label, p))
                return orig(self, *a, **kw)

            return f

        setattr(cls, meth, make())
    yield
    for cls, meth, orig in originals:
        setattr(cls, meth, orig)


class E:
    def embed_documents(self, texts):
        return [[1.0] * 8 for _ in texts]

    def embed_query(self, q):
        return [1.0] * 8

    def get_query_embedding(self, q):
        return [1.0] * 8

    def get_text_embedding(self, t):
        return [1.0] * 8

    def get_agg_embedding_from_queries(self, qs, **kw):
        return [1.0] * 8


TEXTS = ["hello world", "foo bar"]
IDS = ["a", "b"]
VECS = np.ones((2, 8), dtype=np.float32)


def test_langchain():
    from simlar.integrations.langchain.langchain_retriever import SimlarRetriever
    from simlar.integrations.langchain.simlar_vector_store import SimlarVectorStore

    SEEN.clear()
    s = SimlarVectorStore(embedding=E(), parallel=False)
    s.add_texts(TEXTS)
    s.add_texts(["x y"], parallel=True)
    s.similarity_search("hello", k=2)
    s.similarity_search_with_score("hello", k=2, parallel=True)
    SimlarRetriever(vector_store=s, k=2, parallel=True).invoke("hello")
    assert [p for _, p in SEEN] == [False, True, False, True, True], SEEN


def test_haystack():
    from haystack import Document

    from simlar.integrations.haystack.simlar_document_store import SimlarDocumentStore
    from simlar.integrations.haystack.simlar_retriever import SimlarHybridRetriever

    SEEN.clear()
    ds = SimlarDocumentStore(parallel=False)
    ds.write_documents([Document(content=t, embedding=[1.0] * 8) for t in TEXTS])
    ds.write_documents([Document(content="z", embedding=[1.0] * 8)], parallel=True)
    ds.search("hello", [1.0] * 8)
    ds.search("hello", [1.0] * 8, parallel=True)
    SimlarHybridRetriever(ds, top_k=2, parallel=True).run("hello", [1.0] * 8)
    assert [p for _, p in SEEN] == [False, True, False, True, True], SEEN
    # round-trips through to_dict/from_dict
    assert SimlarDocumentStore.from_dict(ds.to_dict())._parallel is False


def test_llamaindex():
    from llama_index.core.vector_stores.types import VectorStoreQuery

    from simlar.integrations.llama_index.simlar_retriever import SimlarRetriever
    from simlar.integrations.llama_index.simlar_vector_store import SimlarVectorStore

    SEEN.clear()
    st = SimlarVectorStore.from_texts(TEXTS, IDS, VECS, parallel=False)
    q = VectorStoreQuery(query_embedding=[1.0] * 8, query_str="hello", similarity_top_k=2)
    st.query(q)
    st.query(q, parallel=True)
    r = SimlarRetriever.from_texts(TEXTS, IDS, VECS, embed_model=E(), k=2, parallel=True)
    r.retrieve("hello")
    assert [p for _, p in SEEN] == [False, False, True, True, True], SEEN
