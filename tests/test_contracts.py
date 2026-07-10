"""Tests for SearchResult, _Parameters (pure Python)."""

from __future__ import annotations

from simlar.contracts import SearchResult


class TestSearchResult:
    def test_fields(self):
        r = SearchResult(rank=1, id="doc_0", score=0.9)
        assert r.rank == 1
        assert r.id == "doc_0"
        assert r.score == 0.9
        assert r.text is None

    def test_with_text(self):
        r = SearchResult(rank=0, id="x", score=1.0, text="hello world")
        assert r.text == "hello world"

    def test_equality(self):
        r1 = SearchResult(rank=0, id="a", score=0.5)
        r2 = SearchResult(rank=0, id="a", score=0.5)
        assert r1 == r2
