"""Tests for simlar._engine: an engine build missing an index core blocks only that index."""

from __future__ import annotations

import importlib
import sys

import pytest

from simlar import IndexUnavailableError, RelevanceIndex, _engine, unavailable_indexes
from simlar.indexes.registry import _REGISTRY, build


class TestEngineCore:
    def test_present_symbol_is_returned(self, monkeypatch):
        monkeypatch.setattr(_engine, "_UNAVAILABLE", {})
        assert _engine.engine_core("x", "json", "dumps") is importlib.import_module("json").dumps
        assert unavailable_indexes() == {}

    @pytest.mark.parametrize(
        ("module", "name"),
        [("simlar_engine.indexes._no_such_impl", "_Core"), ("json", "_NoSuchCore")],
    )
    def test_missing_module_or_symbol_is_recorded(self, monkeypatch, module, name):
        monkeypatch.setattr(_engine, "_UNAVAILABLE", {})
        assert _engine.engine_core("x", module, name) is None
        assert f"{module}.{name}" in unavailable_indexes()["x"]
        with pytest.raises(IndexUnavailableError, match="'x' index is unavailable"):
            _engine.require_core("x", None)

    def test_unavailable_error_is_an_import_error(self):
        assert issubclass(IndexUnavailableError, ImportError)


def test_missing_core_blocks_only_that_index(monkeypatch, tmp_path):
    """Reimport lookup_index with its engine module unimportable, as on an engine build without it."""
    import simlar.indexes.lookup_index as lookup_mod

    monkeypatch.setattr(_engine, "_UNAVAILABLE", {})
    monkeypatch.setitem(_REGISTRY, "lookup", _REGISTRY["lookup"])  # reload re-registers
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "simlar_engine.indexes._hash_match_impl", None)
        importlib.reload(lookup_mod)
    try:
        assert set(unavailable_indexes()) == {"lookup"}
        with pytest.raises(IndexUnavailableError, match=r"_hash_match_impl\._HashMatchCore"):
            lookup_mod.LookupIndex()
        with pytest.raises(IndexUnavailableError):
            lookup_mod.LookupIndex.load(str(tmp_path))
        with pytest.raises(IndexUnavailableError):
            build("lookup")
        assert isinstance(RelevanceIndex(), RelevanceIndex)
    finally:
        importlib.reload(lookup_mod)


def test_engine_without_filtering_gets_stand_ins(monkeypatch, tmp_path):
    """Reimport filtered_index with the engine's filtering types gone, as on a build without them."""
    import simlar_engine

    import simlar.indexes.filtered_index as filtered_mod

    monkeypatch.setattr(_engine, "_UNAVAILABLE", {})
    monkeypatch.setitem(_REGISTRY, "filtered", _REGISTRY["filtered"])  # reload re-registers
    with monkeypatch.context() as m:
        for name in ("FilteredIndex", "SQLFilter", "MetadataFilter", "FilterError"):
            m.delattr(simlar_engine, name, raising=False)
        importlib.reload(filtered_mod)
    try:
        assert "simlar_engine.FilteredIndex" in unavailable_indexes()["filtered"]
        with pytest.raises(IndexUnavailableError, match="'filtered' index is unavailable"):
            filtered_mod.FilteredIndex(RelevanceIndex())
        with pytest.raises(IndexUnavailableError):
            filtered_mod.FilteredIndex.load(str(tmp_path))
        with pytest.raises(IndexUnavailableError):
            build("filtered", inner=RelevanceIndex())
        with pytest.raises(IndexUnavailableError):
            filtered_mod.SQLFilter.where("len >= 6")
        with pytest.raises(IndexUnavailableError):
            filtered_mod.MetadataFilter()
        assert not hasattr(filtered_mod.SQLFilter, "__wrapped__")  # probing stays safe
        with pytest.raises(filtered_mod.FilterError):  # still a real, catchable exception
            raise filtered_mod.FilterError("x")
        assert issubclass(filtered_mod.FilterError, ValueError)
    finally:
        importlib.reload(filtered_mod)
