"""Smoke tests for SimlarDocumentStore and SimlarHybridRetriever (Haystack 2.x)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("haystack", reason="haystack-ai not installed")

from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError

from simlar.integrations.haystack.simlar_document_store import SimlarDocumentStore
from simlar.integrations.haystack.simlar_retriever import SimlarHybridRetriever

DIM = 8
UNIT_VEC: list[float] = np.ones(DIM, dtype=np.float32).tolist()


def _doc(content: str, doc_id: str | None = None) -> Document:
    kwargs = {"content": content, "embedding": UNIT_VEC}
    if doc_id:
        kwargs["id"] = doc_id
    return Document(**kwargs)


def _doc_meta(content: str, meta: dict, doc_id: str | None = None) -> Document:
    kwargs = {"content": content, "embedding": UNIT_VEC, "meta": meta}
    if doc_id:
        kwargs["id"] = doc_id
    return Document(**kwargs)


@pytest.fixture()
def store():
    return SimlarDocumentStore(top_k=5)


@pytest.fixture()
def populated_store(store):
    store.write_documents([_doc("cancer treatment", "doc_0"), _doc("machine learning", "doc_1")])
    return store


class TestSimlarDocumentStore:
    def test_empty_store_count(self, store):
        assert store.count_documents() == 0

    def test_write_returns_count(self, store):
        n = store.write_documents([_doc("hello"), _doc("world")])
        assert n == 2

    def test_count_after_write(self, store):
        store.write_documents([_doc("a"), _doc("b")])
        assert store.count_documents() == 2

    def test_write_without_embedding_raises(self, store):
        with pytest.raises(ValueError, match="no embedding"):
            store.write_documents([Document(content="no embedding")])

    def test_empty_search_returns_empty(self, store):
        assert store.search("query", UNIT_VEC) == []

    def test_search_returns_documents(self, populated_store):
        results = populated_store.search("cancer", UNIT_VEC, top_k=2)
        assert len(results) >= 1
        assert all(hasattr(d, "content") for d in results)

    def test_search_result_has_meta_score(self, populated_store):
        results = populated_store.search("query", UNIT_VEC, top_k=1)
        assert "score" in results[0].meta

    def test_duplicate_skip_policy(self, store):
        doc = _doc("hello", "x")
        store.write_documents([doc])
        n = store.write_documents([doc], policy=DuplicatePolicy.SKIP)
        assert n == 0
        assert store.count_documents() == 1

    def test_duplicate_fail_policy(self, store):
        doc = _doc("hello", "x")
        store.write_documents([doc])
        with pytest.raises(DuplicateDocumentError):
            store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_filter_documents_returns_all(self, populated_store):
        assert len(populated_store.filter_documents()) == 2

    def test_to_dict_from_dict_roundtrip(self, store):
        d = store.to_dict()
        loaded = SimlarDocumentStore.from_dict(d)
        assert loaded._top_k == store._top_k

    def test_write_empty_list(self, store):
        assert store.write_documents([]) == 0


class TestDuplicatePolicies:
    def test_default_none_policy_overwrites(self, store):
        doc = _doc("hello", "x")
        store.write_documents([doc])
        # NONE policy tombstones the old position and writes a new one
        store.write_documents([_doc("hello again", "x")])
        assert store.count_documents() == 1


class TestSearchTombstonesAndFilters:
    def test_search_skips_deleted(self, store):
        store.write_documents([_doc("alpha", "d0"), _doc("beta", "d1")])
        store.delete_documents(["d0"])
        results = store.search("alpha", UNIT_VEC, top_k=5)
        assert all(r.meta["doc_id"] != 0 for r in results)

    def test_search_applies_filters(self, store):
        store.write_documents(
            [
                _doc_meta("alpha", {"lang": "en"}, "d0"),
                _doc_meta("beta", {"lang": "fr"}, "d1"),
            ]
        )
        filters = {"operator": "==", "field": "meta.lang", "value": "fr"}
        results = store.search("alpha", UNIT_VEC, top_k=5, filters=filters)
        assert all(r.meta.get("lang") == "fr" for r in results)


class TestDeletion:
    def test_delete_documents(self, populated_store):
        populated_store.delete_documents(["doc_0"])
        assert populated_store.count_documents() == 1

    def test_delete_unknown_id_is_noop(self, populated_store):
        populated_store.delete_documents(["does_not_exist"])
        assert populated_store.count_documents() == 2

    def test_delete_all_documents(self, populated_store):
        populated_store.delete_all_documents()
        assert populated_store.count_documents() == 0
        # store is usable again after reset
        assert populated_store.write_documents([_doc("fresh")]) == 1

    def test_delete_by_filter(self, store):
        store.write_documents(
            [
                _doc_meta("alpha", {"lang": "en"}, "d0"),
                _doc_meta("beta", {"lang": "fr"}, "d1"),
            ]
        )
        deleted = store.delete_by_filter({"operator": "==", "field": "meta.lang", "value": "en"})
        assert deleted == 1
        assert store.count_documents() == 1


class TestFilterAndCount:
    def test_filter_documents_with_filters(self, store):
        store.write_documents([_doc_meta("a", {"lang": "en"}), _doc_meta("b", {"lang": "fr"})])
        matched = store.filter_documents({"operator": "==", "field": "meta.lang", "value": "en"})
        assert len(matched) == 1

    def test_count_documents_by_filter(self, store):
        store.write_documents([_doc_meta("a", {"lang": "en"}), _doc_meta("b", {"lang": "en"})])
        n = store.count_documents_by_filter({"operator": "==", "field": "meta.lang", "value": "en"})
        assert n == 2


class TestUpdate:
    def test_update_by_filter(self, store):
        store.write_documents([_doc_meta("a", {"lang": "en", "reviewed": False})])
        n = store.update_by_filter(
            {"operator": "==", "field": "meta.lang", "value": "en"},
            {"reviewed": True},
        )
        assert n == 1
        assert store.filter_documents()[0].meta["reviewed"] is True


class TestMetadataIntrospection:
    def test_get_metadata_fields_info(self, store):
        store.write_documents(
            [_doc_meta("a", {"name": "x", "count": 3, "flag": True, "ratio": 1.5})]
        )
        info = store.get_metadata_fields_info()
        assert info["name"]["type"] == "keyword"
        assert info["count"]["type"] == "long"
        assert info["flag"]["type"] == "boolean"
        assert info["ratio"]["type"] == "float"

    def test_get_metadata_field_min_max(self, store):
        store.write_documents([_doc_meta("a", {"count": 3}), _doc_meta("b", {"count": 7})])
        assert store.get_metadata_field_min_max("count") == {"min": 3, "max": 7}

    def test_get_metadata_field_min_max_empty(self, store):
        assert store.get_metadata_field_min_max("count") == {"min": None, "max": None}

    def test_get_metadata_field_unique_values(self, store):
        store.write_documents(
            [
                _doc_meta("a", {"lang": "en"}),
                _doc_meta("b", {"lang": "en"}),
                _doc_meta("c", {"lang": "fr"}),
            ]
        )
        assert set(store.get_metadata_field_unique_values("lang")) == {"en", "fr"}

    def test_count_unique_metadata_by_filter(self, store):
        store.write_documents(
            [
                _doc_meta("a", {"lang": "en", "topic": "x"}),
                _doc_meta("b", {"lang": "fr", "topic": "x"}),
            ]
        )
        counts = store.count_unique_metadata_by_filter(
            {"operator": "!=", "field": "meta.lang", "value": "de"},
            ["lang", "topic"],
        )
        assert counts == {"lang": 2, "topic": 1}


class TestStorePersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        store = SimlarDocumentStore(top_k=5)
        store.write_documents(
            [_doc_meta("alpha", {"lang": "en"}, "d0"), _doc_meta("beta", {"lang": "fr"}, "d1")]
        )
        store.delete_documents(["d0"])

        directory = str(tmp_path / "store")
        store.save(directory)

        loaded = SimlarDocumentStore.load(directory)
        assert loaded.count_documents() == 1
        assert loaded._top_k == 5
        assert loaded.filter_documents()[0].meta["lang"] == "fr"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No saved store"):
            SimlarDocumentStore.load(str(tmp_path / "nonexistent"))


class TestFilterDSL:
    def _doc(self):
        return _doc_meta("body text", {"lang": "en", "count": 5, "tags": {"a": 1}}, "d0")

    def test_missing_operator_and_conditions_raises(self, store):
        with pytest.raises(FilterError, match="operator"):
            store._check_condition(self._doc(), {"field": "meta.lang"})

    def test_missing_field_raises(self, store):
        with pytest.raises(FilterError, match="field"):
            store._check_condition(self._doc(), {"operator": "==", "value": "en"})

    def test_field_not_string_raises(self, store):
        with pytest.raises(FilterError, match="must be a string"):
            store._check_condition(self._doc(), {"operator": "==", "field": 123, "value": "en"})

    def test_missing_value_raises(self, store):
        with pytest.raises(FilterError, match="value"):
            store._check_condition(self._doc(), {"operator": "==", "field": "meta.lang"})

    def test_equality_operators(self, store):
        doc = self._doc()
        assert store._check_condition(doc, {"operator": "==", "field": "meta.lang", "value": "en"})
        assert store._check_condition(doc, {"operator": "!=", "field": "meta.lang", "value": "fr"})

    def test_in_operators(self, store):
        doc = self._doc()
        assert store._check_condition(
            doc, {"operator": "in", "field": "meta.lang", "value": ["en", "fr"]}
        )
        assert store._check_condition(
            doc, {"operator": "not in", "field": "meta.lang", "value": ["de", "fr"]}
        )

    def test_in_requires_list(self, store):
        with pytest.raises(FilterError, match="must be a list"):
            store._check_condition(
                self._doc(), {"operator": "in", "field": "meta.lang", "value": "en"}
            )

    def test_not_in_requires_list(self, store):
        with pytest.raises(FilterError, match="must be a list"):
            store._check_condition(
                self._doc(), {"operator": "not in", "field": "meta.lang", "value": "en"}
            )

    def test_comparison_operators(self, store):
        doc = self._doc()
        assert store._check_condition(doc, {"operator": ">", "field": "meta.count", "value": 3})
        assert store._check_condition(doc, {"operator": ">=", "field": "meta.count", "value": 5})
        assert store._check_condition(doc, {"operator": "<", "field": "meta.count", "value": 9})
        assert store._check_condition(doc, {"operator": "<=", "field": "meta.count", "value": 5})

    def test_comparison_none_returns_false(self, store):
        assert not store._check_condition(
            self._doc(), {"operator": ">", "field": "meta.missing", "value": 3}
        )

    def test_comparison_type_mismatch_raises(self, store):
        with pytest.raises(FilterError, match="Type mismatch"):
            store._check_condition(self._doc(), {"operator": ">", "field": "meta.lang", "value": 3})

    def test_comparison_type_error_raises(self, store):
        # Two same-typed but non-comparable values (dicts) trigger the TypeError branch
        with pytest.raises(FilterError, match="Type mismatch"):
            store._check_condition(
                self._doc(), {"operator": ">", "field": "meta.tags", "value": {"b": 2}}
            )

    def test_unknown_operator_returns_false(self, store):
        assert not store._check_condition(
            self._doc(), {"operator": "LIKE", "field": "meta.lang", "value": "en"}
        )

    def test_and_or_not(self, store):
        doc = self._doc()
        assert store._check_condition(
            doc,
            {
                "operator": "AND",
                "conditions": [
                    {"operator": "==", "field": "meta.lang", "value": "en"},
                    {"operator": ">", "field": "meta.count", "value": 1},
                ],
            },
        )
        assert store._check_condition(
            doc,
            {
                "operator": "OR",
                "conditions": [
                    {"operator": "==", "field": "meta.lang", "value": "fr"},
                    {"operator": "==", "field": "meta.lang", "value": "en"},
                ],
            },
        )
        assert store._check_condition(
            doc,
            {
                "operator": "NOT",
                "conditions": [{"operator": "==", "field": "meta.lang", "value": "fr"}],
            },
        )

    def test_and_missing_conditions_raises(self, store):
        with pytest.raises(FilterError, match="conditions"):
            store._check_condition(self._doc(), {"operator": "AND"})

    def test_or_missing_conditions_raises(self, store):
        with pytest.raises(FilterError, match="conditions"):
            store._check_condition(self._doc(), {"operator": "OR"})

    def test_not_missing_conditions_raises(self, store):
        with pytest.raises(FilterError, match="conditions"):
            store._check_condition(self._doc(), {"operator": "NOT"})

    def test_not_empty_conditions_raises(self, store):
        with pytest.raises(FilterError, match="at least one condition"):
            store._check_condition(self._doc(), {"operator": "NOT", "conditions": []})

    def test_get_doc_value_content_and_id(self, store):
        doc = self._doc()
        assert store._get_doc_value(doc, "content") == "body text"
        assert store._get_doc_value(doc, "id") == "d0"

    def test_get_doc_value_attribute_and_fallback(self, store):
        doc = self._doc()
        # "score" is a real Document attribute → hasattr branch
        assert store._get_doc_value(doc, "score") == doc.score
        # bare key not an attribute → meta fallback
        assert store._get_doc_value(doc, "lang") == "en"


class TestSimlarHybridRetriever:
    def test_run_returns_documents_key(self, populated_store):
        retriever = SimlarHybridRetriever(document_store=populated_store)
        out = retriever.run(query="cancer", query_embedding=UNIT_VEC)
        assert "documents" in out
        assert isinstance(out["documents"], list)

    def test_run_empty_store_returns_empty(self, store):
        retriever = SimlarHybridRetriever(document_store=store)
        out = retriever.run(query="query", query_embedding=UNIT_VEC)
        assert out["documents"] == []

    def test_run_top_k_override(self, populated_store):
        retriever = SimlarHybridRetriever(document_store=populated_store, top_k=10)
        out = retriever.run(query="cancer", query_embedding=UNIT_VEC, top_k=1)
        assert len(out["documents"]) <= 1
