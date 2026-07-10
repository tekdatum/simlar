"""Tests for HelixIndex (wraps the stubbed _HelixCore from conftest)."""

from __future__ import annotations

import numpy as np

from simlar.indexes.helix_index import HelixIndex


def _make_index() -> HelixIndex:
    index = HelixIndex(top_k=10)
    index.add(ids=["a", "b", "c"], texts=["alpha", "beta", "gamma"])
    return index


class TestAddAndSearch:
    def test_add_populates_index(self):
        index = _make_index()
        assert index.size == 3
        assert index.is_trained is True

    def test_search_returns_results(self):
        index = _make_index()
        results = index.search(query_text="alpha", k=2)
        assert [r.id for r in results] == ["a", "b"]

    def test_search_defaults_to_top_k(self):
        index = _make_index()
        results = index.search(query_text="alpha")
        assert len(results) == 3


class TestFit:
    def test_fit_without_params(self):
        index = HelixIndex()
        index.fit(corpus=["alpha", "beta"], vectors=np.zeros((2, 4), dtype=np.float32))
        assert index.is_trained is True

    def test_fit_with_params_kwarg(self):
        index = HelixIndex()
        params = object()
        index.fit(
            corpus=["alpha", "beta"],
            vectors=np.zeros((2, 4), dtype=np.float32),
            params=params,
        )
        assert index.is_trained is True


class TestMetadata:
    def test_index_type(self):
        assert _make_index().index_type == "helix"

    def test_boundaries_and_fit_values_default_none(self):
        index = _make_index()
        assert index.boundaries is None
        assert index.fit_values is None

    def test_params_property(self):
        assert _make_index()._params is None


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        index = _make_index()
        directory = str(tmp_path / "helix")
        index.save(directory)

        loaded = HelixIndex.load(directory)
        assert isinstance(loaded, HelixIndex)
        assert loaded.index_type == "helix"
