"""Tests for RelevanceIndex (wraps the stubbed _RelevanceCore from conftest)."""

from __future__ import annotations

import numpy as np

from simlar.indexes.relevance_index import RelevanceIndex


def _make_index() -> RelevanceIndex:
    index = RelevanceIndex()
    index.add(["a", "b", "c"], ["hello world", "foo bar", "baz qux"])
    return index


class TestConstruction:
    def test_default_construction(self):
        index = RelevanceIndex()
        assert index.size == 0
        assert index.is_trained is False

    def test_custom_params(self):
        index = RelevanceIndex(
            method="lucene", k1=1.2, b=0.5, stopwords_lang="french", stemmer_lang="french"
        )
        assert index.size == 0


class TestMutations:
    def test_add_populates_index(self):
        index = _make_index()
        assert index.size == 3
        assert index.is_trained is True

    def test_fit(self):
        index = RelevanceIndex()
        index.fit(["hello world", "foo bar"])
        assert index.is_trained is True

    def test_update(self):
        index = _make_index()
        index.update(["a"], ["updated text"])
        assert index.size == 3

    def test_delete(self):
        index = _make_index()
        index.delete(["a"])
        assert index.size == 3  # stub delete is a no-op


class TestSearch:
    def test_search_returns_results(self):
        index = _make_index()
        results = index.search("hello", k=2)
        assert [r.id for r in results] == ["a", "b"]

    def test_search_raw(self):
        index = _make_index()
        ids, scores = index.search_raw("hello", k=2)
        assert isinstance(ids, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert len(ids) == 2


class TestMetadata:
    def test_index_type(self):
        assert _make_index().index_type == "relevance"


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        index = _make_index()
        directory = str(tmp_path / "relevance")
        index.save(directory)

        loaded = RelevanceIndex.load(directory)
        assert isinstance(loaded, RelevanceIndex)
        assert loaded.index_type == "relevance"
