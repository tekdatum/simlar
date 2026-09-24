"""Tests for BM25CIndex (backed by the external bm25c library - no simlar_engine core
involved, and no bm25s either).

Two behaviors worth knowing, confirmed empirically rather than assumed:
  - bm25c/BM25CRelevanceCore mirrors _RelevanceCore's own behavior: duplicate ids
    within/across add() calls are silently allowed (creates two entries), not rejected.
  - An untrained (never fit()/add()'d) index's search()/search_raw() raises via the k > n_docs
    check ("k (N) exceeds the number of indexed documents (0)"), not an "untrained" message.

Corpora here use enough documents that the terms under test are genuinely rare (appear in fewer
than half the documents) - bm25c's Robertson-variant idf clamps to exactly 0 for a term in >=
half the corpus (ln(1) when the ratio hits exactly 1, same mechanism as the classic "more than
half" case), so a term split evenly across a small corpus contributes nothing and every match
comes back as a padding slot (id=None) instead of a real result.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "bm25c", reason="bm25c not installed -- pip install -e <bm25c repo>/python first"
)

from simlar.indexes.bm25c_index import BM25CIndex


def _make_index() -> BM25CIndex:
    index = BM25CIndex()
    index.add(
        ["a", "b", "c", "d", "e", "f"],
        [
            "fox jumps quick",
            "dog sleeps lazy",
            "cat runs fast",
            "bird flies high",
            "fish swims deep",
            "wolf howls loud",
        ],
    )
    return index


class TestConstruction:
    def test_default_construction(self):
        index = BM25CIndex()
        assert index.size == 0
        assert index.is_trained is False

    def test_custom_params(self):
        index = BM25CIndex(k1=1.2, b=0.5)
        assert index.size == 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            BM25CIndex(method="not_a_real_method")

    def test_invalid_stopwords_lang_raises(self):
        with pytest.raises(ValueError, match="stopwords_lang"):
            BM25CIndex(stopwords_lang="spanish")

    def test_invalid_stemmer_lang_raises(self):
        with pytest.raises(ValueError, match="stemmer_lang"):
            BM25CIndex(stemmer_lang="spanish")


class TestMutations:
    def test_add_populates_index(self):
        index = _make_index()
        assert index.size == 6
        assert index.is_trained is True
        assert index.ids == ["a", "b", "c", "d", "e", "f"]

    def test_fit(self):
        index = BM25CIndex()
        index.fit(["fox jumps quick", "dog sleeps lazy"])
        assert index.is_trained is True
        assert index.ids == ["0", "1"]

    def test_fit_empty_corpus_raises(self):
        index = BM25CIndex()
        with pytest.raises(ValueError, match="empty"):
            index.fit([])

    def test_add_duplicate_id_silently_allowed(self):
        # Matches _RelevanceCore's own documented behavior -
        # BM25CRelevanceCore was deliberately designed to mirror _RelevanceCore here.
        index = _make_index()
        index.add(["a"], ["already present"])
        assert index.size == 7
        assert index.ids.count("a") == 2

    def test_update(self):
        index = _make_index()
        index.update(["a"], ["zebra migration patterns"])
        assert index.size == 6
        results = index.search("zebra", k=6)
        real = [r for r in results if r.id is not None]
        assert [r.id for r in real] == ["a"]

    def test_update_unknown_id_raises(self):
        index = _make_index()
        with pytest.raises(ValueError, match="not found"):
            index.update(["nope"], ["text"])

    def test_delete(self):
        index = _make_index()
        index.delete(["b"])
        assert index.size == 5
        assert index.ids == ["a", "c", "d", "e", "f"]

    def test_delete_unknown_id_raises(self):
        index = _make_index()
        with pytest.raises(ValueError, match="not found"):
            index.delete(["nope"])

    def test_delete_then_search_confirms_document_gone(self):
        index = _make_index()
        index.delete(["b"])
        # 'b' ("dog sleeps lazy") is gone - searching its unique term returns no real match.
        results = index.search("sleeps", k=5)
        assert all(r.id is None for r in results)


class TestSearch:
    def test_search_returns_results(self):
        index = _make_index()
        results = index.search("fox", k=6)
        real = [r for r in results if r.id is not None]
        assert [r.id for r in real] == ["a"]

    def test_search_on_untrained_index_raises(self):
        index = BM25CIndex()
        with pytest.raises(ValueError, match="exceeds the number of indexed documents"):
            index.search("anything", k=1)

    def test_search_raw_single_query(self):
        index = _make_index()
        ids, scores = index.search_raw("fox", k=3)
        assert isinstance(ids, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert ids.dtype == np.int64
        assert scores.dtype == np.float64
        assert ids.shape == (1, 3)
        # only 'a' (position 0) matches "fox" - the rest of the row is padding (-1)
        assert ids[0, 0] == 0
        assert (ids[0, 1:] == -1).all()

    def test_search_raw_batch_queries(self):
        index = _make_index()
        ids, scores = index.search_raw(["fox", "cat"], k=2)
        assert ids.shape == (2, 2)
        assert scores.shape == (2, 2)

    def test_search_raw_on_untrained_index_raises(self):
        index = BM25CIndex()
        with pytest.raises(ValueError, match="exceeds the number of indexed documents"):
            index.search_raw("anything", k=1)


def _make_index_with_shared_term() -> BM25CIndex:
    # "swim(s)" matches both c and e (after stemming) -- needed to prove
    # candidates EXCLUDES a real match, not just one that never mattered.
    index = BM25CIndex()
    index.add(
        ["a", "b", "c", "d", "e", "f"],
        [
            "fox jumps quick",
            "dog sleeps lazy",
            "cat swims fast",
            "bird flies high",
            "fish swims deep",
            "wolf howls loud",
        ],
    )
    return index


class TestCandidates:
    def test_search_raw_restricts_to_candidates(self):
        index = _make_index_with_shared_term()
        ids, _ = index.search_raw("swim", k=6)
        assert set(ids[0][ids[0] != -1].tolist()) == {2, 4}  # c, e

        ids, _ = index.search_raw("swim", k=6, candidates=np.array([2]))
        assert set(ids[0][ids[0] != -1].tolist()) == {2}

    def test_search_raw_candidates_excludes_real_match(self):
        index = _make_index_with_shared_term()
        ids, _ = index.search_raw("swim", k=6, candidates=np.array([0, 1, 3, 5]))
        assert (ids == -1).all()

    def test_search_raw_candidates_scores_match_unrestricted(self):
        index = _make_index_with_shared_term()
        ids, scores = index.search_raw("swim", k=6)
        unrestricted = {int(i): float(s) for i, s in zip(ids[0], scores[0], strict=True) if i != -1}

        ids, scores = index.search_raw("swim", k=6, candidates=np.array([2, 4]))
        restricted = {int(i): float(s) for i, s in zip(ids[0], scores[0], strict=True) if i != -1}
        assert restricted == unrestricted

    def test_search_raw_candidates_shared_across_batch(self):
        index = _make_index_with_shared_term()
        ids, _ = index.search_raw(["swim", "fox"], k=6, candidates=np.array([0, 2]))
        assert set(ids[0][ids[0] != -1].tolist()) == {2}  # "swim": e excluded
        assert set(ids[1][ids[1] != -1].tolist()) == {0}  # "fox": still matches a

    def test_search_candidates_forwards_from_search(self):
        index = _make_index_with_shared_term()
        results = index.search("swim", k=6, candidates=np.array([2]))
        real = {r.id for r in results if r.id is not None}
        assert real == {"c"}


class TestMetadata:
    def test_index_type(self):
        assert _make_index().index_type == "bm25c"


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        index = _make_index()
        directory = str(tmp_path / "bm25c")
        index.save(directory)

        loaded = BM25CIndex.load(directory)
        assert isinstance(loaded, BM25CIndex)
        assert loaded.index_type == "bm25c"
        assert loaded.ids == ["a", "b", "c", "d", "e", "f"]
        results = loaded.search("fox", k=6)
        real = [r for r in results if r.id is not None]
        assert [r.id for r in real] == ["a"]

    def test_save_and_load_round_trip_after_mutation(self, tmp_path):
        index = _make_index()
        index.update(["a"], ["zebra migration"])
        index.delete(["b"])
        directory = str(tmp_path / "bm25c_mutated")
        index.save(directory)

        loaded = BM25CIndex.load(directory)
        assert loaded.ids == ["a", "c", "d", "e", "f"]
        results = loaded.search("zebra", k=5)
        real = [r for r in results if r.id is not None]
        assert [r.id for r in real] == ["a"]


class TestImportGuard:
    def test_missing_bm25c_raises_clear_import_error(self, monkeypatch):
        import simlar.indexes.bm25c_index as mod

        monkeypatch.setattr(mod, "bm25c", None)
        with pytest.raises(ImportError, match="bm25c"):
            mod.BM25CIndex()


class TestHelixIntegration:
    def test_helix_index_with_bm25c_text_index(self):
        from simlar import HelixIndex

        bm = BM25CIndex()
        helix = HelixIndex(text_index=bm)
        corpus = [
            "fox jumps quick",
            "dog sleeps lazy",
            "cat runs fast",
            "bird flies high",
            "fish swims deep",
            "wolf howls loud",
        ]
        vectors = np.random.default_rng(0).standard_normal((6, 8)).astype(np.float32)
        helix.add(["a", "b", "c", "d", "e", "f"], texts=corpus, vectors=vectors)

        assert helix.text_index is bm
        assert bm.size == 6
        assert bm.ids == ["a", "b", "c", "d", "e", "f"]
        results = [r.id for r in bm.search("fox", k=6) if r.id is not None]
        assert results == ["a"]


class TestStreamingHelixIntegration:
    def test_streaming_helix_index_accepts_bm25c_class(self):
        from simlar import StreamingHybridIndex

        streaming = StreamingHybridIndex(text_index_cls=BM25CIndex)
        corpus = [
            "fox jumps quick",
            "dog sleeps lazy",
            "cat runs fast",
            "bird flies high",
            "fish swims deep",
            "wolf howls loud",
        ]
        vectors = np.random.default_rng(0).standard_normal((6, 8)).astype(np.float32)
        streaming.add_batch(corpus, vectors)

        ids, scores = streaming.search("fox", vectors[0], k=3)
        assert isinstance(ids, np.ndarray)
        assert isinstance(scores, np.ndarray)
