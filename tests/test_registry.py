"""Tests for register, build, load_from_directory (ported from private repo)."""

from __future__ import annotations

import pytest

from simlar.indexes.registry import _REGISTRY, build, load_from_directory, register
from simlar.persistence import write_config


class _Dummy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def load(cls, directory: str) -> _Dummy:
        return cls()


class TestRegisterAndBuild:
    def test_register_and_build_roundtrip(self):
        register("_test_dummy_register")(_Dummy)
        obj = build("_test_dummy_register", foo="bar")
        assert isinstance(obj, _Dummy)
        assert obj.kwargs == {"foo": "bar"}

    def test_register_returns_same_class(self):
        result = register("_test_identity")(_Dummy)
        assert result is _Dummy

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown index type"):
            build("_nonexistent_index_xyz")

    def test_register_overwrites_existing(self):
        register("_test_overwrite")(_Dummy)

        class _Other:
            def __init__(self, **kwargs):
                pass

        register("_test_overwrite")(_Other)
        assert isinstance(build("_test_overwrite"), _Other)

    def test_registered_name_appears_in_registry(self):
        register("_test_in_registry")(_Dummy)
        assert "_test_in_registry" in _REGISTRY


class TestLoadFromDirectory:
    def test_unknown_type_raises(self, tmp_path):
        write_config(tmp_path / "config.json", {"index_type": "_never_registered_xyz"})
        with pytest.raises(KeyError):
            load_from_directory(str(tmp_path))

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_from_directory(str(tmp_path))

    def test_load_calls_cls_load(self, tmp_path):
        loaded = []

        class _Tracking:
            def __init__(self, **kwargs):
                pass

            @classmethod
            def load(cls, directory):
                loaded.append(directory)
                return cls()

        register("_test_load_dir")(_Tracking)
        write_config(tmp_path / "config.json", {"index_type": "_test_load_dir"})
        obj = load_from_directory(str(tmp_path))
        assert isinstance(obj, _Tracking)
        assert loaded == [str(tmp_path)]
