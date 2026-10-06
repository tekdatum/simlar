"""Tests for pluggable embedders (simlar.embeddings) and the indexes' embedder= support."""

from __future__ import annotations

import sys
import zlib

import numpy as np
import pytest

from simlar import CallableEmbedder, Embedder, FilteredIndex, HelixIndex, SimlarEngine
from simlar.embeddings import _embed_queries, as_embedder
from simlar.indexes.lookup_index import LookupIndex
from simlar.indexes.registry import load_from_directory
from simlar.indexes.relevance_index import RelevanceIndex

REAL_ENGINE = getattr(sys.modules.get("simlar_engine"), "__file__", None) is not None
real_engine = pytest.mark.skipif(
    not REAL_ENGINE, reason="needs the real engine (SIMLAR_REAL_ENGINE=1)"
)
stub_engine = pytest.mark.skipif(REAL_ENGINE, reason="inspects the stub engine's calls")

DIM = 16
IDS = ["a", "b", "c", "d"]
TEXTS = ["apple pie recipe", "car engine repair", "chocolate cake baking", "bicycle tire repair"]


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words embeddings: shared words -> nearby vectors."""
    out = np.full((len(texts), DIM), 0.01, dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            out[row, zlib.crc32(word.encode()) % DIM] += 1.0
    return out / np.linalg.norm(out, axis=1, keepdims=True)


class _ListEmbeddings:
    """LangChain-shaped embedder returning plain lists."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts):
        return _hash_embed(list(texts)).tolist()

    def embed_query(self, text):
        self.queries.append(text)
        return _hash_embed([text])[0].tolist()


class _Spy:
    """Wraps an index core and records the calls made to it."""

    def __init__(self, core) -> None:
        self._core = core
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        attr = getattr(self._core, name)
        if not callable(attr):
            return attr

        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return attr(*args, **kwargs)

        return record

    def last(self, name):
        return next(c for c in reversed(self.calls) if c[0] == name)


def _spy(index) -> _Spy:
    index._core = _Spy(index._core)
    return index._core


class TestCallableEmbedder:
    def test_shape_and_dtype(self):
        out = CallableEmbedder(lambda t: [[1, 2]] * len(t)).embed_documents(["x", "y", "z"])
        assert out.shape == (3, 2)
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]

    def test_normalize(self):
        e = CallableEmbedder(lambda t: [[3.0, 4.0]] * len(t), normalize=True)
        np.testing.assert_allclose(e.embed_documents(["x"]), [[0.6, 0.8]], rtol=1e-6)
        np.testing.assert_allclose(e.embed_query("x"), [0.6, 0.8], rtol=1e-6)

    def test_normalize_keeps_zero_vectors(self):
        e = CallableEmbedder(lambda t: [[0.0, 0.0]] * len(t), normalize=True)
        assert not np.isnan(e.embed_documents(["x"])).any()

    def test_batching(self):
        sizes: list[int] = []

        def fn(texts):
            sizes.append(len(texts))
            return _hash_embed(texts)

        out = CallableEmbedder(fn, batch_size=2).embed_documents(TEXTS + ["extra"])
        assert sizes == [2, 2, 1]
        np.testing.assert_allclose(out, _hash_embed(TEXTS + ["extra"]))

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            CallableEmbedder(_hash_embed, batch_size=0)

    def test_query_defaults_to_fn(self):
        np.testing.assert_allclose(
            CallableEmbedder(_hash_embed).embed_query("apple"), _hash_embed(["apple"])[0]
        )

    def test_query_fn(self):
        e = CallableEmbedder(_hash_embed, query_fn=lambda q: np.ones(DIM))
        np.testing.assert_array_equal(e.embed_query("apple"), np.ones(DIM, dtype=np.float32))

    def test_row_count_mismatch(self):
        with pytest.raises(ValueError, match="2 vectors for 3 texts"):
            CallableEmbedder(lambda t: np.zeros((2, 4))).embed_documents(["x", "y", "z"])

    def test_wrong_rank(self):
        with pytest.raises(ValueError, match="expected"):
            CallableEmbedder(lambda t: np.zeros((2, 2, 2))).embed_documents(["x", "y"])

    def test_is_an_embedder(self):
        assert isinstance(CallableEmbedder(_hash_embed), Embedder)


class TestAsEmbedder:
    def test_none(self):
        assert as_embedder(None) is None

    def test_protocol_object_passes_through(self):
        e = _ListEmbeddings()
        assert as_embedder(e) is e

    def test_callable_is_wrapped(self):
        assert isinstance(as_embedder(_hash_embed), CallableEmbedder)

    def test_invalid(self):
        with pytest.raises(TypeError):
            as_embedder(42)

    def test_query_batch(self):
        e = _ListEmbeddings()
        assert _embed_queries(e, "apple").shape == (DIM,)
        assert _embed_queries(e, ["apple", "car"]).shape == (2, DIM)
        assert e.queries == ["apple", "apple", "car"]


@stub_engine
class TestSimlarEngine:
    def test_add_texts(self):
        idx = SimlarEngine(embedder=_hash_embed)
        spy = _spy(idx)
        idx.add(IDS, texts=TEXTS)
        _, args, _ = spy.last("add")
        assert args[0] == IDS
        np.testing.assert_allclose(args[1], _hash_embed(TEXTS))
        assert args[1].dtype == np.float32

    def test_update_and_fit_texts(self):
        idx = SimlarEngine(embedder=_hash_embed)
        spy = _spy(idx)
        idx.fit(texts=TEXTS)
        np.testing.assert_allclose(spy.last("fit")[1][0], _hash_embed(TEXTS))
        idx.update(["a"], texts=["new text"])
        np.testing.assert_allclose(spy.last("update")[1][1], _hash_embed(["new text"]))

    def test_search_query_text(self):
        idx = SimlarEngine(embedder=_ListEmbeddings())
        idx.add(IDS, texts=TEXTS)
        spy = _spy(idx)
        idx.search(query_text="apple", k=2)
        query = spy.last("search")[1][0]
        assert query.shape == (DIM,)
        np.testing.assert_allclose(query, _hash_embed(["apple"])[0], rtol=1e-6)

    def test_vectors_still_work(self):
        idx = SimlarEngine(embedder=_hash_embed)
        spy = _spy(idx)
        vecs = np.ones((4, DIM), dtype=np.float32)
        idx.add(IDS, vecs)
        assert spy.last("add")[1][1] is vecs

    def test_both_raises(self):
        idx = SimlarEngine(embedder=_hash_embed)
        with pytest.raises(ValueError, match="not both"):
            idx.add(IDS, np.ones((4, DIM), dtype=np.float32), texts=TEXTS)
        with pytest.raises(ValueError, match="not both"):
            idx.search(np.ones(DIM, dtype=np.float32), query_text="x")

    def test_no_embedder_raises(self):
        idx = SimlarEngine()
        with pytest.raises(ValueError, match="no embedder"):
            idx.add(IDS, texts=TEXTS)
        with pytest.raises(ValueError, match="no embedder"):
            idx.search(query_text="x")
        with pytest.raises(ValueError, match="required"):
            idx.add(IDS)

    def test_embedder_property(self):
        idx = SimlarEngine()
        assert idx.embedder is None
        idx.embedder = _hash_embed
        assert isinstance(idx.embedder, CallableEmbedder)


@stub_engine
class TestHelixIndex:
    def test_add_texts_feeds_both_sides(self):
        idx = HelixIndex(embedder=_hash_embed)
        spy = _spy(idx)
        idx.add(IDS, TEXTS)
        _, (ids, texts, vectors, _parallel), _ = spy.last("add")
        assert texts == TEXTS
        np.testing.assert_allclose(vectors, _hash_embed(TEXTS))

    def test_update_reembeds(self):
        idx = HelixIndex(embedder=_hash_embed)
        idx.add(IDS, TEXTS)
        spy = _spy(idx)
        idx.update(["a"], ["new text"])
        np.testing.assert_allclose(spy.last("update")[1][2], _hash_embed(["new text"]))

    def test_fit_embeds_corpus(self):
        idx = HelixIndex(embedder=_hash_embed)
        spy = _spy(idx)
        idx.fit(TEXTS)
        np.testing.assert_allclose(spy.last("fit")[1][1], _hash_embed(TEXTS))

    def test_fit_without_embedder_or_vectors_raises(self):
        with pytest.raises(ValueError, match="no embedder"):
            HelixIndex().fit(TEXTS)

    def test_search_str_is_single_query(self):
        idx = HelixIndex(embedder=_hash_embed)
        idx.add(IDS, TEXTS)
        spy = _spy(idx)
        idx.search("apple", k=2)
        assert spy.last("search")[1][1].shape == (DIM,)

    def test_search_list_is_batch(self):
        idx = HelixIndex(embedder=_hash_embed)
        idx.add(IDS, TEXTS)
        spy = _spy(idx)
        idx.search(["apple", "car"], k=2)
        assert spy.last("search")[1][1].shape == (2, DIM)

    def test_explicit_query_vector_wins(self):
        idx = HelixIndex(embedder=_hash_embed)
        idx.add(IDS, TEXTS)
        spy = _spy(idx)
        qv = np.ones(DIM, dtype=np.float32)
        idx.search("apple", qv, k=2)
        assert spy.last("search")[1][1] is qv

    # LlamaIndex/Haystack build HelixIndex without an embedder and rely on
    # text-only searches staying BM25-only.
    def test_no_embedder_text_only_search_unchanged(self):
        idx = HelixIndex()
        idx.add(IDS, TEXTS)
        spy = _spy(idx)
        idx.search(query_text="apple", k=2)
        assert spy.last("search")[1][1] is None

    def test_no_embedder_add_texts_only(self):
        idx = HelixIndex()
        spy = _spy(idx)
        idx.add(IDS, TEXTS)
        assert spy.last("add")[1][2] is None


@stub_engine
class TestPersistence:
    def test_simlar_load_with_embedder(self, tmp_path):
        idx = SimlarEngine(embedder=_hash_embed)
        idx.add(IDS, texts=TEXTS)
        idx.save(str(tmp_path))
        assert SimlarEngine.load(str(tmp_path)).embedder is None
        assert SimlarEngine.load(str(tmp_path), embedder=_hash_embed).embedder is not None

    def test_helix_loaded_without_embedder(self, tmp_path):
        idx = HelixIndex(embedder=_hash_embed)
        idx.add(IDS, TEXTS)
        idx.save(str(tmp_path))
        assert HelixIndex.load(str(tmp_path)).embedder is None

    @pytest.mark.parametrize("cls", [SimlarEngine, HelixIndex])
    def test_load_from_directory(self, tmp_path, cls):
        cls().save(str(tmp_path))
        e = _ListEmbeddings()
        assert load_from_directory(str(tmp_path), embedder=e).embedder is e
        assert load_from_directory(str(tmp_path)).embedder is None

    def test_load_from_directory_filtered(self, tmp_path):
        FilteredIndex(HelixIndex()).save(str(tmp_path))
        e = _ListEmbeddings()
        loaded = FilteredIndex.load(str(tmp_path), embedder=e)
        assert loaded.inner.embedder is e
        assert loaded.embedder is e

    def test_load_from_directory_text_index_rejects(self, tmp_path):
        LookupIndex().save(str(tmp_path))
        with pytest.raises(TypeError, match="does not take an embedder"):
            load_from_directory(str(tmp_path), embedder=_hash_embed)


@stub_engine
class TestFilteredIndex:
    def test_add_and_search_text_only(self):
        e = _ListEmbeddings()
        idx = FilteredIndex(HelixIndex(embedder=e))
        spy = _spy(idx.inner)
        idx.add(IDS, TEXTS, metadata=[{"n": i} for i in range(4)])
        np.testing.assert_allclose(spy.last("add")[1][2], _hash_embed(TEXTS), rtol=1e-6)
        assert idx.search(query_text="apple", k=2)
        assert e.queries == ["apple"]

    def test_embedder_setter_delegates(self):
        idx = FilteredIndex(SimlarEngine())
        idx.embedder = _hash_embed
        assert isinstance(idx.inner.embedder, CallableEmbedder)

    def test_embedder_setter_rejects_text_index(self):
        with pytest.raises(TypeError):
            FilteredIndex(RelevanceIndex()).embedder = _hash_embed


@real_engine
class TestRealEngine:
    def test_helix_text_only_roundtrip(self, tmp_path):
        idx = FilteredIndex(HelixIndex(embedder=_hash_embed))
        idx.add(IDS, TEXTS, metadata=[{"kind": "food" if i % 2 == 0 else "fix"} for i in range(4)])
        assert idx.search(query_text="chocolate cake", k=1)[0].id == "c"
        hits = idx.search(query_text="repair", k=4, filter="kind = 'fix'")
        assert {r.id for r in hits} == {"b", "d"}

        idx.save(str(tmp_path))
        loaded = load_from_directory(str(tmp_path), embedder=_hash_embed)
        before = [(r.id, r.score) for r in idx.search(query_text="apple pie", k=3)]
        after = [(r.id, r.score) for r in loaded.search(query_text="apple pie", k=3)]
        assert before == after

    def test_simlar_engine_texts(self):
        idx = SimlarEngine(embedder=_hash_embed)
        idx.add(IDS, texts=TEXTS)
        assert idx.search(query_text="car engine", k=1)[0].id == "b"
        idx.update(["b"], texts=["strawberry jam"])
        assert idx.search(query_text="strawberry jam", k=1)[0].id == "b"
