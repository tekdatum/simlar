"""Tests for write_config / read_config (pure Python)."""

from __future__ import annotations

import json

from simlar.persistence import FORMAT_VERSION, read_config, write_config


class TestWriteConfig:
    def test_writes_json(self, tmp_path):
        p = tmp_path / "config.json"
        write_config(p, {"index_type": "simlar"})
        data = json.loads(p.read_text())
        assert data["index_type"] == "simlar"
        assert data["format_version"] == FORMAT_VERSION

    def test_extra_keys_preserved(self, tmp_path):
        p = tmp_path / "config.json"
        write_config(p, {"a": 1, "b": [1, 2]})
        data = json.loads(p.read_text())
        assert data["a"] == 1
        assert data["b"] == [1, 2]

    def test_format_version_injected(self, tmp_path):
        p = tmp_path / "config.json"
        write_config(p, {})
        data = json.loads(p.read_text())
        assert data["format_version"] == FORMAT_VERSION


class TestReadConfig:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "config.json"
        write_config(p, {"index_type": "bm25"})
        data = read_config(p)
        assert data["index_type"] == "bm25"

    def test_format_version_present(self, tmp_path):
        p = tmp_path / "config.json"
        write_config(p, {})
        data = read_config(p)
        assert "format_version" in data
