"""Tests for BM25xIndex (backed by the real bm25x package - no simlar_engine core involved).

Unlike test_relevance_index.py, this exercises the real bm25x library end-to-end, per the task's
acceptance criteria ("CI execution using the real bm25x package"). bm25x only ships wheels for
Python 3.12 (see BM25xIndex's module docstring), so this whole file skips cleanly on other
interpreters rather than failing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bm25x", reason="bm25x only ships wheels for Python 3.12")

from simlar.indexes.bm25x_index import BM25xIndex


def _make_index() -> BM25xIndex:
    index = BM25xIndex()
    index.add(["a", "b", "c"], ["fox jumps quick", "dog sleeps lazy", "cat runs fast"])
    return index


class TestConstruction:
    def test_default_construction(self):
        index = BM25xIndex()
        assert index.size == 0
        assert index.is_trained is False

    def test_custom_params(self):
        index = BM25xIndex(method="robertson", k1=1.2, b=0.5, delta=1.0)
        assert index.size == 0

    def test_invalid_method_propagates_library_error(self):
        with pytest.raises(ValueError, match="method"):
            BM25xIndex(method="not_a_real_method")


class TestMutations:
    def test_add_populates_index(self):
        index = _make_index()
        assert index.size == 3
        assert index.is_trained is True
        assert index.ids == ["a", "b", "c"]

    def test_fit(self):
        index = BM25xIndex()
        index.fit(["fox jumps quick", "dog sleeps lazy"])
        assert index.is_trained is True
        assert index.ids == ["0", "1"]

    def test_fit_empty_corpus_raises(self):
        index = BM25xIndex()
        with pytest.raises(ValueError, match="empty"):
            index.fit([])

    def test_add_duplicate_id_raises(self):
        index = _make_index()
        with pytest.raises(ValueError, match="duplicate"):
            index.add(["a"], ["already present"])

    def test_add_duplicate_within_batch_raises(self):
        index = BM25xIndex()
        with pytest.raises(ValueError, match="duplicate"):
            index.add(["x", "x"], ["one", "two"])

    def test_update(self):
        index = _make_index()
        index.update(["a"], ["zebra migration patterns"])
        assert index.size == 3
        results = index.search("zebra", k=3)
        assert [r.id for r in results] == ["a"]

    def test_update_unknown_id_raises(self):
        index = _make_index()
        with pytest.raises(ValueError, match="unknown"):
            index.update(["nope"], ["text"])

    def test_delete(self):
        index = _make_index()
        index.delete(["b"])
        assert index.size == 2
        assert index.ids == ["a", "c"]

    def test_delete_unknown_id_raises(self):
        index = _make_index()
        with pytest.raises(ValueError, match="unknown"):
            index.delete(["nope"])

    def test_delete_then_search_confirms_document_gone(self):
        index = _make_index()
        index.delete(["b"])
        # 'b' ("dog sleeps lazy") is gone - searching its unique term returns nothing.
        assert index.search("sleeps", k=3) == []

    def test_delete_compacts_positions_correctly(self):
        # bm25x.delete() shifts subsequent documents' internal integer positions down (verified
        # directly against the real library while planning this) - this locks in that BM25xIndex's
        # own id<->position bookkeeping tracks that compaction correctly, not just that the ids
        # list happens to look right.
        index = BM25xIndex()
        index.add(
            ["w", "x", "y", "z"],
            ["alpha unique", "bravo unique", "charlie unique", "delta unique"],
        )
        index.delete(["x"])  # was position 1; 'y' and 'z' should now be at positions 1 and 2
        assert index.ids == ["w", "y", "z"]
        assert [r.id for r in index.search("charlie", k=1)] == ["y"]
        assert [r.id for r in index.search("delta", k=1)] == ["z"]
        assert index.search("bravo", k=1) == []


class TestSearch:
    def test_search_returns_results(self):
        index = _make_index()
        results = index.search("fox", k=3)
        assert [r.id for r in results] == ["a"]

    def test_search_on_untrained_index_raises(self):
        index = BM25xIndex()
        with pytest.raises(ValueError, match="untrained"):
            index.search("anything", k=1)

    def test_search_raw_single_query(self):
        index = _make_index()
        ids, scores = index.search_raw("fox", k=3)
        assert isinstance(ids, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert ids.dtype == np.int64
        assert scores.dtype == np.float64
        assert ids.shape == (3,)
        # only 'a' matches "fox" - the rest of the row is padded with -1
        assert ids[0] == 0
        assert (ids[1:] == -1).all()

    def test_search_raw_batch_queries(self):
        index = _make_index()
        ids, scores = index.search_raw(["fox", "cat"], k=2)
        assert ids.shape == (2, 2)
        assert scores.shape == (2, 2)

    def test_search_raw_on_untrained_index_raises(self):
        index = BM25xIndex()
        with pytest.raises(ValueError, match="untrained"):
            index.search_raw("anything", k=1)


class TestMetadata:
    def test_index_type(self):
        assert _make_index().index_type == "bm25x"


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        index = _make_index()
        directory = str(tmp_path / "bm25x")
        index.save(directory)

        loaded = BM25xIndex.load(directory)
        assert isinstance(loaded, BM25xIndex)
        assert loaded.index_type == "bm25x"
        assert loaded.ids == ["a", "b", "c"]
        assert [r.id for r in loaded.search("fox", k=3)] == ["a"]

    def test_save_and_load_round_trip_after_mutation(self, tmp_path):
        index = _make_index()
        index.update(["a"], ["zebra migration"])
        index.delete(["b"])
        directory = str(tmp_path / "bm25x_mutated")
        index.save(directory)

        loaded = BM25xIndex.load(directory)
        assert loaded.ids == ["a", "c"]
        assert [r.id for r in loaded.search("zebra", k=2)] == ["a"]


class TestImportGuard:
    def test_missing_bm25x_raises_clear_import_error(self, monkeypatch):
        import simlar.indexes.bm25x_index as mod

        monkeypatch.setattr(mod, "bm25x", None)
        with pytest.raises(ImportError, match="bm25x"):
            mod.BM25xIndex()


class TestHelixIntegration:
    def test_helix_index_with_bm25x_text_index(self):
        from simlar import HelixIndex

        bm = BM25xIndex()
        helix = HelixIndex(text_index=bm)
        corpus = ["fox jumps quick", "dog sleeps lazy", "cat runs fast", "fox and dog play"]
        vectors = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
        helix.add(["a", "b", "c", "d"], texts=corpus, vectors=vectors)

        assert helix.text_index is bm
        assert bm.size == 4
        assert bm.ids == ["a", "b", "c", "d"]
        assert [r.id for r in bm.search("fox", k=4)] == ["a", "d"]


class TestStreamingHelixIntegration:
    def test_streaming_helix_index_accepts_bm25x_class(self):
        from simlar import StreamingHybridIndex

        streaming = StreamingHybridIndex(text_index_cls=BM25xIndex)
        corpus = ["fox jumps quick", "dog sleeps lazy", "cat runs fast", "fox and dog play"]
        vectors = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
        streaming.add_batch(corpus, vectors)

        ids, scores = streaming.search("fox", vectors[0], k=3)
        assert isinstance(ids, np.ndarray)
        assert isinstance(scores, np.ndarray)
