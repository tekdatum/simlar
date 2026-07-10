"""Tests for SimlarEngine (wraps the stubbed _SimlarCore from conftest)."""

from __future__ import annotations

import numpy as np

from simlar.contracts import _Parameters
from simlar.indexes.simlar_engine import SimlarEngine


def _make_index() -> SimlarEngine:
    index = SimlarEngine()
    index.add(["a", "b", "c"], np.zeros((3, 4), dtype=np.float32))
    return index


class TestConstruction:
    def test_default_construction(self):
        index = SimlarEngine()
        assert index.size == 0
        assert index.is_trained is False

    def test_custom_n_candidates(self):
        index = SimlarEngine(n_candidates=100)
        assert index.size == 0


class TestMutations:
    def test_add_populates_index(self):
        index = _make_index()
        assert index.size == 3
        assert index.is_trained is True

    def test_fit(self):
        index = SimlarEngine()
        index.fit(np.zeros((2, 4), dtype=np.float32))
        assert index.is_trained is True

    def test_update(self):
        index = _make_index()
        index.update(["a"], np.ones((1, 4), dtype=np.float32))
        assert index.size == 3

    def test_delete(self):
        index = _make_index()
        index.delete(["a"])
        assert index.size == 3  # stub delete is a no-op


class TestSearch:
    def test_search_returns_results(self):
        index = _make_index()
        results = index.search(np.zeros(4, dtype=np.float32), k=2)
        assert [r.id for r in results] == ["a", "b"]

    def test_search_raw(self):
        index = _make_index()
        ids, scores = index.search_raw(np.zeros((1, 4), dtype=np.float32), k=2)
        assert isinstance(ids, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert len(ids) == 2


class TestMetadata:
    def test_index_type(self):
        assert _make_index().index_type == "simlar"

    def test_coreindex(self):
        assert _make_index().coreindex is None

    def test_matrix(self):
        assert _make_index()._matrix is None

    def test_params_none_when_unfitted(self):
        index = _make_index()
        assert index._params is None
        assert index.boundaries is None
        assert index.fit_values is None

    def test_boundaries_and_fit_values_from_params(self):
        index = SimlarEngine()
        boundaries = np.array([0.0, 1.0], dtype=np.float32)
        fit_values = np.array([2.0, 3.0], dtype=np.float32)
        index.fit(
            np.zeros((2, 4), dtype=np.float32),
            params=_Parameters(boundaries=boundaries, fit_values=fit_values),
        )
        assert index._params is not None
        np.testing.assert_array_equal(index.boundaries, boundaries)
        np.testing.assert_array_equal(index.fit_values, fit_values)


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        index = _make_index()
        directory = str(tmp_path / "simlar")
        index.save(directory)

        loaded = SimlarEngine.load(directory)
        assert isinstance(loaded, SimlarEngine)
        assert loaded.index_type == "simlar"
