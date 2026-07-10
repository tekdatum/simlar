"""Tests for LookupIndex (wraps the stubbed _TextCore from conftest)."""

from __future__ import annotations

import numpy as np

from simlar.indexes.lookup_index import LookupIndex


def _make_index() -> LookupIndex:
    index = LookupIndex()
    index.add(["a", "b", "c"], ["hello world", "foo bar", "baz qux"])
    return index


class TestConstruction:
    def test_default_construction(self):
        index = LookupIndex()
        assert index.size == 0
        assert index.is_trained is False

    def test_custom_langs(self):
        index = LookupIndex(stopwords_lang="french", stemmer_lang="french")
        assert index.size == 0


class TestMutations:
    def test_add_populates_index(self):
        index = _make_index()
        assert index.size == 3
        assert index.is_trained is True

    def test_fit(self):
        index = LookupIndex()
        index.fit(["hello world", "foo bar"])
        assert index.is_trained is True

    def test_update(self):
        index = _make_index()
        index.update(["a"], ["updated text"])
        assert index.size == 3

    def test_delete(self):
        index = _make_index()
        index.delete(["a"])
        assert index.size == 2


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
        assert _make_index().index_type == "lookup"


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        index = _make_index()
        directory = str(tmp_path / "lookup")
        index.save(directory)

        loaded = LookupIndex.load(directory)
        assert isinstance(loaded, LookupIndex)
        assert loaded.index_type == "lookup"
