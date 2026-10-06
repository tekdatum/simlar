"""Tests for the Haystack 3.x SimlarDocumentStore and retrievers.

The first group runs everywhere (engine stubs or the real engine). Filtering,
roles, ranking and Haystack's own DocumentStore test suite need the real
engine: run with ``SIMLAR_REAL_ENGINE=1``.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("haystack", reason="haystack-ai not installed")

from haystack import Document, Pipeline
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy, FilterPolicy
from haystack.errors import FilterError
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    create_filterable_docs,
)
from haystack.utils.filters import document_matches_filter

import simlar.integrations.haystack.simlar_document_store as store_module
from simlar.integrations.haystack.simlar_document_store import SimlarDocumentStore
from simlar.integrations.haystack.simlar_retriever import (
    SimlarBM25Retriever,
    SimlarEmbeddingRetriever,
    SimlarHybridRetriever,
)

REAL_ENGINE = getattr(sys.modules.get("simlar_engine"), "__file__", None) is not None
real_engine = pytest.mark.skipif(
    not REAL_ENGINE, reason="needs the real engine (SIMLAR_REAL_ENGINE=1)"
)

DIM = 8
_RNG = np.random.default_rng(0)


def _vec() -> list[float]:
    return _RNG.random(DIM).astype(np.float32).tolist()


def _doc(content: str, doc_id: str | None = None, **meta) -> Document:
    kwargs = {"content": content, "embedding": _vec(), "meta": meta}
    if doc_id:
        kwargs["id"] = doc_id
    return Document(**kwargs)


def _ids(docs) -> list[str]:
    return [d.id for d in docs]


_CORPUS = [
    ("apple pie recipe baking", {"topic": "food", "year": 2019, "lang": "en"}),
    ("sourdough bread baking", {"topic": "food", "year": 2021, "lang": "fr"}),
    ("car engine repair", {"topic": "cars", "year": 2020, "lang": "en"}),
    ("electric car battery", {"topic": "cars", "year": 2023, "lang": "de"}),
    ("neural network training", {"topic": "ml", "year": 2022}),
    ("gradient boosting trees", {"topic": "ml", "year": 2018, "lang": "en"}),
]
_ROLES = [["public"], ["public"], ["staff"], ["staff"], ["public", "staff"], ["staff"]]


@pytest.fixture()
def store():
    return SimlarDocumentStore(parallel=False)


@pytest.fixture()
def corpus_store(store):
    docs = [_doc(text, f"d{i}", **meta) for i, (text, meta) in enumerate(_CORPUS)]
    store.write_documents(docs, roles=_ROLES)
    return store, docs


# ── Runs with the stubs too ───────────────────────────────────────────────────


class TestBasics:
    def test_empty(self, store):
        assert store.count_documents() == 0
        assert store.filter_documents() == []
        assert store.search("query", _vec()) == []
        assert store.embedding_retrieval(_vec()) == []

    def test_write_returns_count(self, store):
        assert store.write_documents([_doc("hello"), _doc("world")]) == 2
        assert store.count_documents() == 2

    def test_write_empty_list(self, store):
        assert store.write_documents([]) == 0

    def test_missing_embedding_rejected(self, store):
        with pytest.raises(ValueError, match="no embedding"):
            store.write_documents([_doc("fine"), Document(content="no embedding")])
        assert store.count_documents() == 0  # nothing written

    def test_invalid_input(self, store):
        with pytest.raises(ValueError):
            store.write_documents(["not a document"])  # type: ignore[list-item]
        with pytest.raises(ValueError):
            store.write_documents("not a list")  # type: ignore[arg-type]

    def test_dimension_mismatch_rejected(self, store):
        store.write_documents([_doc("apple")])
        with pytest.raises(ValueError, match="dimension"):
            store.write_documents([Document(content="banana", embedding=[0.1, 0.2])])

    def test_roles_length_checked(self, store):
        with pytest.raises(ValueError, match="roles length"):
            store.write_documents([_doc("apple")], roles=[["x"], ["y"]])

    def test_duplicate_policies(self, store):
        store.write_documents([_doc("hello", "x")])
        assert store.write_documents([_doc("again", "x")], policy=DuplicatePolicy.SKIP) == 0
        with pytest.raises(DuplicateDocumentError):
            store.write_documents([_doc("again", "x")], policy=DuplicatePolicy.FAIL)
        # NONE overwrites, as before.
        assert store.write_documents([_doc("replaced", "x")]) == 1
        assert store.count_documents() == 1
        assert store.filter_documents()[0].content == "replaced"

    def test_fail_policy_is_atomic(self, store):
        store.write_documents([_doc("apple", "x")])
        with pytest.raises(DuplicateDocumentError):
            store.write_documents([_doc("new", "y"), _doc("dup", "x")], policy=DuplicatePolicy.FAIL)
        assert _ids(store.filter_documents()) == ["x"]

    def test_delete(self, store):
        store.write_documents([_doc("apple", "a"), _doc("banana", "b")])
        store.delete_documents(["a", "missing"])
        assert _ids(store.filter_documents()) == ["b"]
        store.delete_documents(["b"])  # the last one
        assert store.count_documents() == 0
        assert store.write_documents([_doc("cherry", "c")]) == 1

    def test_store_to_dict_roundtrip(self):
        store = SimlarDocumentStore(top_k=7, relevance_k=11, core_k=13, parallel=False)
        data = store.to_dict()
        assert data["type"].endswith("simlar_document_store.SimlarDocumentStore")
        loaded = SimlarDocumentStore.from_dict(data)
        assert loaded.to_dict() == data

    def test_results_carry_score_not_meta(self, store):
        store.write_documents([_doc("alpha", "a", lang="en")])
        (doc,) = store.search("alpha", _vec(), top_k=1)
        assert doc.id == "a"
        assert doc.score is not None
        assert doc.meta == {"lang": "en"}
        assert doc.embedding is None
        (doc,) = store.hybrid_retrieval("alpha", _vec(), top_k=1, return_embedding=True)
        assert doc.embedding is not None

    def test_returned_documents_dont_alias_store(self, store):
        store.write_documents([_doc("alpha", "a", lang="en")])
        store.filter_documents()[0].meta["lang"] = "fr"
        assert store.filter_documents()[0].meta["lang"] == "en"


class TestRetrieverComponents:
    @pytest.mark.parametrize(
        "cls", [SimlarEmbeddingRetriever, SimlarBM25Retriever, SimlarHybridRetriever]
    )
    def test_to_dict_from_dict(self, store, cls):
        retriever = cls(
            store,
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
            top_k=3,
            filter_policy=FilterPolicy.MERGE,
            roles=["staff"],
        )
        data = retriever.to_dict()
        assert data["init_parameters"]["filter_policy"] == "merge"
        loaded = cls.from_dict(data)
        assert isinstance(loaded.document_store, SimlarDocumentStore)
        assert loaded.filter_policy == FilterPolicy.MERGE
        assert (loaded.filters, loaded.top_k, loaded.roles) == (retriever.filters, 3, ["staff"])

    def test_rejects_other_store_and_bad_top_k(self, store):
        with pytest.raises(TypeError):
            SimlarEmbeddingRetriever(document_store=object())  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            SimlarBM25Retriever(store, top_k=0)

    def test_hybrid_keeps_legacy_signature(self, store):
        store.write_documents([_doc("cancer treatment", "a"), _doc("machine learning", "b")])
        retriever = SimlarHybridRetriever(document_store=store)
        assert retriever.top_k == 5
        out = retriever.run(query="cancer", query_embedding=_vec(), top_k=1)
        assert len(out["documents"]) == 1

    def test_pipeline_dumps_loads(self, store):
        pipeline = Pipeline()
        pipeline.add_component("retriever", SimlarHybridRetriever(store, top_k=2))
        loaded = Pipeline.loads(pipeline.dumps())
        retriever = loaded.get_component("retriever")
        assert isinstance(retriever, SimlarHybridRetriever)
        assert retriever.top_k == 2


# ── Real engine: Haystack's DocumentStore test suite ──────────────────────────


class _AutoEmbedStore(SimlarDocumentStore):
    """Test-only: gives embedding-less documents a random embedding, so
    Haystack's suite (whose documents mostly have none) can run. The real
    store rejects them."""

    def write_documents(self, documents, policy=DuplicatePolicy.NONE, roles=None, parallel=None):
        if isinstance(documents, list):
            documents = [
                d
                if not isinstance(d, Document) or d.embedding is not None
                else replace(d, embedding=[random.random() for _ in range(DIM)])
                for d in documents
            ]
        return super().write_documents(documents, policy, roles, parallel)


@real_engine
class TestHaystackDocumentStoreSuite(
    DocumentStoreBaseExtendedTests,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    @pytest.fixture
    def document_store(self):
        return _AutoEmbedStore(parallel=False)

    @pytest.fixture
    def filterable_docs(self):
        # Haystack's documents use 768-dim embeddings; keep one dimension per store.
        return [
            replace(d, embedding=None if d.embedding is None else d.embedding[:DIM])
            for d in create_filterable_docs()
        ]

    @staticmethod
    def assert_documents_are_equal(received, expected):
        # Embeddings are random (see _AutoEmbedStore); compare everything else.
        strip = lambda docs: [replace(d, embedding=None) for d in docs]  # noqa: E731
        assert strip(received) == strip(expected)

    def test_write_documents(self, document_store):
        docs = [Document(id="1", content="first"), Document(id="2", content="second")]
        assert document_store.write_documents(docs) == 2
        # DuplicatePolicy.NONE overwrites.
        assert document_store.write_documents([Document(id="1", content="again")]) == 1
        self.assert_documents_are_equal(
            document_store.filter_documents(),
            [Document(id="1", content="again"), Document(id="2", content="second")],
        )


# ── Real engine: filter semantics vs. Haystack's reference implementation ────


def _random_meta(rng: random.Random) -> dict:
    meta: dict = {}
    choices = {
        "n": lambda: rng.randint(-3, 3),
        "x": lambda: rng.choice([-1.5, 0.0, 0.5, 2.0, 2]),
        "flag": lambda: rng.choice([True, False]),
        "s": lambda: rng.choice(["a", "b", "c", "A"]),
        "mixed": lambda: rng.choice([1, "1", 2.0, "x"]),
        "date": lambda: rng.choice(
            ["1969-07-21T20:17:40", "1972-12-11 19:54:58", "1989-11-09T17:53:00+00:00"]
        ),
        "tags": lambda: rng.sample(["a", "b", "c"], rng.randint(0, 2)),
        "nested": lambda: {"k": rng.randint(0, 2)},
        "file-name": lambda: rng.choice(["f1", "f2"]),
        "position": lambda: rng.randint(0, 5),
        "big": lambda: rng.choice([2**70, 1]),
    }
    for key, make in choices.items():
        roll = rng.random()
        if roll < 0.15:
            meta[key] = None
        elif roll < 0.75:
            meta[key] = make()
    return meta


_VALUES = [
    None,
    0,
    1,
    2,
    -1.5,
    0.5,
    True,
    False,
    "a",
    "A",
    "1",
    "x",
    "f1",
    "1970-01-01T00:00:00",
    "1972-12-11T19:54:58",
    2**70,
]
_FIELDS = [
    "meta.n",
    "meta.x",
    "meta.flag",
    "meta.s",
    "meta.mixed",
    "meta.date",
    "meta.tags",
    "meta.nested",
    "meta.nested.k",
    "meta.file-name",
    "meta.position",
    "meta.big",
    "meta.never",
    "n",
    "s",
    "id",
    "content",
]


def _random_filter(rng: random.Random, depth: int = 0) -> dict:
    if depth < 2 and rng.random() < 0.35:
        op = rng.choice(["AND", "OR", "NOT"])
        return {
            "operator": op,
            "conditions": [_random_filter(rng, depth + 1) for _ in range(rng.randint(1, 3))],
        }
    op = rng.choice(["==", "!=", ">", ">=", "<", "<=", "in", "not in"])
    if op in ("in", "not in"):
        value = rng.sample(_VALUES, rng.randint(0, 3))
    else:
        value = rng.choice(_VALUES)
    return {"field": rng.choice(_FIELDS), "operator": op, "value": value}


@real_engine
class TestFilterSemantics:
    @pytest.fixture(scope="class")
    def randomized(self):
        rng = random.Random(1234)
        docs = [_doc(f"doc {i}", f"id{i}", **_random_meta(rng)) for i in range(80)]
        store = SimlarDocumentStore(parallel=False)
        store.write_documents(docs[:60])
        # Overwrites and deletes must keep SQL and documents in step.
        store.write_documents(
            [replace(d, meta=_random_meta(rng)) for d in docs[40:60]],
            policy=DuplicatePolicy.OVERWRITE,
        )
        store.write_documents(docs[60:])
        store.delete_documents([f"id{i}" for i in range(0, 80, 7)])
        store.update_by_filter(
            {"field": "meta.s", "operator": "==", "value": "a"}, {"s": ["a", "z"]}
        )
        return store, store.filter_documents()

    def test_matches_reference_on_random_filters(self, randomized):
        store, docs = randomized
        rng = random.Random(99)
        checked = 0
        for _ in range(1500):
            filters = _random_filter(rng)
            try:
                expected = [d.id for d in docs if document_matches_filter(filters, d)]
            except Exception:
                continue  # an error case; covered by Haystack's suite
            assert _ids(store.filter_documents(filters)) == expected, filters
            checked += 1
        assert checked > 800

    def test_reference_errors(self, corpus_store):
        store, _ = corpus_store
        bad = [
            {"field": "meta.year", "operator": "in", "value": 2020},
            {"field": "meta.year", "operator": ">", "value": [1]},
            {"field": "meta.year", "operator": ">", "value": "2020"},
            {"field": "meta.year", "operator": "LIKE", "value": "x"},
            {"operator": "XOR", "conditions": []},
            {"conditions": []},
            {"field": "meta.year", "value": 1},
        ]
        for filters in bad:
            with pytest.raises(FilterError):
                store.filter_documents(filters)

    def test_exact_filters_stay_in_sql(self, corpus_store, monkeypatch):
        store, _ = corpus_store
        calls = []
        real = store_module.document_matches_filter
        monkeypatch.setattr(
            store_module,
            "document_matches_filter",
            lambda f, d: calls.append(1) or real(f, d),
        )
        exact = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.year", "operator": ">=", "value": 2019},
                {"field": "meta.lang", "operator": "not in", "value": ["de", None]},
                {
                    "operator": "NOT",
                    "conditions": [
                        {"field": "meta.topic", "operator": "==", "value": "ml"},
                    ],
                },
                {"field": "id", "operator": "!=", "value": "d1"},
            ],
        }
        assert _ids(store.filter_documents(exact)) == ["d0", "d2"]
        assert calls == []
        # `content` isn't in SQL: narrowed there, then rechecked.
        content = {"field": "content", "operator": "==", "value": "car engine repair"}
        assert _ids(store.filter_documents(content)) == ["d2"]
        assert calls

    def test_awkward_metadata_keys(self, store):
        store.write_documents(
            [
                _doc("apple", "a", position=7, Name="x", big=2**70, nested={"k": 1}, group="g"),
                _doc("banana", "b", name="y", position=1, order=2),
                _doc("cherry", "c"),
            ]
        )
        # "position" must not have moved any row: the index and filter agree.
        assert store.index.ids == ["a", "b", "c"]
        assert store.index.sql_filter.ids == ["a", "b", "c"]
        by = lambda f, op, v: _ids(store.filter_documents({"field": f, "operator": op, "value": v}))  # noqa: E731
        assert by("meta.position", "==", 7) == ["a"]
        assert by("meta.Name", "==", "x") == ["a"]
        assert by("meta.name", "==", "y") == ["b"]
        assert by("meta.big", "==", 2**70) == ["a"]
        assert by("meta.nested.k", "==", 1) == ["a"]
        assert by("meta.group", "==", "g") == ["a"]
        assert by("meta.order", ">", 1) == ["b"]


# ── Real engine: retrieval ────────────────────────────────────────────────────


@real_engine
class TestRetrieval:
    @pytest.fixture(scope="class")
    def big(self):
        rng = np.random.default_rng(7)
        words = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
        docs = [
            Document(
                id=f"d{i}",
                content=" ".join(rng.choice(words, 4)),
                embedding=rng.random(DIM).astype(np.float32).tolist(),
                meta={"n": int(i % 10), "group": "rare" if i % 25 == 0 else "common"},
            )
            for i in range(300)
        ]
        store = SimlarDocumentStore(parallel=False)
        store.write_documents(docs)
        return store, docs

    def test_filtered_embedding_topk_is_exact(self, big):
        store, docs = big
        q = docs[3].embedding
        filters = {"field": "meta.n", "operator": "in", "value": [3, 4]}
        everything = store.embedding_retrieval(q, top_k=len(docs))
        expected = [d.id for d in everything if d.meta["n"] in (3, 4)][:10]
        assert _ids(store.embedding_retrieval(q, filters=filters, top_k=10)) == expected

    def test_selective_filter_still_returns_k(self, big):
        store, docs = big
        rare = {"field": "meta.group", "operator": "==", "value": "rare"}
        allowed = {d.id for d in docs if d.meta["group"] == "rare"}  # 12 docs
        results = store.embedding_retrieval(docs[0].embedding, filters=rare, top_k=10)
        assert len(results) == 10
        assert set(_ids(results)) <= allowed
        # Hybrid cascades BM25 -> vector, so under a filter it returns only the
        # allowed documents the BM25 stage matched (engine behaviour).
        results = store.hybrid_retrieval("alpha gamma", docs[0].embedding, filters=rare, top_k=10)
        assert 0 < len(results) <= 10
        assert set(_ids(results)) <= allowed

    def test_embedding_retrieval_not_capped_by_pool(self, big):
        store, docs = big  # core_k is 50
        assert len(store.embedding_retrieval(docs[0].embedding, top_k=120)) == 120

    def test_bm25(self, corpus_store):
        store, _ = corpus_store
        results = store.bm25_retrieval("sourdough", top_k=5)
        assert _ids(results) == ["d1"]
        assert results[0].score > 0
        filters = {"field": "meta.topic", "operator": "==", "value": "cars"}
        assert set(_ids(store.bm25_retrieval("car", filters=filters))) == {"d2", "d3"}
        assert store.bm25_retrieval("sourdough", filters=filters) == []

    def test_roles(self, corpus_store):
        store, docs = corpus_store
        q = docs[0].embedding
        public = set(_ids(store.embedding_retrieval(q, top_k=10, roles=["public"])))
        assert public == {"d0", "d1", "d4"}
        assert store.embedding_retrieval(q, top_k=10, roles=[]) == []
        assert set(_ids(store.filter_documents(roles=["staff"]))) == {"d2", "d3", "d4", "d5"}
        store.grant(["d2"], ["public"])
        store.revoke(["d4"], ["public"])
        assert set(_ids(store.embedding_retrieval(q, top_k=10, roles=["public"]))) == {
            "d0",
            "d1",
            "d2",
        }

    def test_overwrite_in_place(self, corpus_store):
        store, docs = corpus_store
        store.write_documents(
            [
                Document(
                    id="d1",
                    content="kayak paddle",
                    embedding=docs[1].embedding,
                    meta={"topic": "boats"},
                )
            ]
        )
        assert store.index.size == len(docs)
        assert store.bm25_retrieval("sourdough") == []
        assert _ids(store.bm25_retrieval("kayak")) == ["d1"]
        # Dropped keys are cleared: d1 no longer has a year or lang.
        assert "d1" not in _ids(
            store.filter_documents({"field": "meta.year", "operator": ">", "value": 0})
        )
        assert "d1" in _ids(
            store.filter_documents({"field": "meta.lang", "operator": "==", "value": None})
        )

    def test_delete_compacts(self, corpus_store):
        store, docs = corpus_store
        store.delete_by_filter({"field": "meta.topic", "operator": "==", "value": "food"})
        assert store.index.ids == ["d2", "d3", "d4", "d5"]
        results = store.embedding_retrieval(docs[2].embedding, top_k=10, roles=["staff"])
        assert set(_ids(results)) == {"d2", "d3", "d4", "d5"}

    def test_retrievers_with_filter_policy(self, corpus_store):
        store, docs = corpus_store
        init = {"field": "meta.topic", "operator": "==", "value": "food"}
        runtime = {"field": "meta.lang", "operator": "==", "value": "en"}
        q = docs[0].embedding
        replace_ = SimlarEmbeddingRetriever(store, filters=init, top_k=10)
        assert set(_ids(replace_.run(q)["documents"])) == {"d0", "d1"}
        assert set(_ids(replace_.run(q, filters=runtime)["documents"])) == {"d0", "d2", "d5"}
        merge = SimlarEmbeddingRetriever(store, filters=init, top_k=10, filter_policy="merge")
        assert _ids(merge.run(q, filters=runtime)["documents"]) == ["d0"]
        assert _ids(SimlarBM25Retriever(store, roles=["staff"]).run("car")["documents"]) != []

    def test_hybrid_pipeline(self, corpus_store):
        store, docs = corpus_store
        pipeline = Pipeline()
        pipeline.add_component("retriever", SimlarHybridRetriever(store, top_k=3))
        out = pipeline.run(
            {
                "retriever": {
                    "query": "car engine",
                    "query_embedding": docs[2].embedding,
                    "filters": {"field": "meta.year", "operator": ">=", "value": 2020},
                    "roles": ["staff"],
                }
            }
        )
        results = out["retriever"]["documents"]
        assert results[0].id == "d2"
        assert set(_ids(results)) <= {"d2", "d3", "d4"}


# ── Real engine: persistence ──────────────────────────────────────────────────


@real_engine
class TestPersistence:
    def test_round_trip(self, corpus_store, tmp_path):
        store, docs = corpus_store
        store.delete_documents(["d0"])
        store.save(tmp_path / "s")
        loaded = SimlarDocumentStore.load(tmp_path / "s")
        assert loaded.to_dict() == store.to_dict()
        assert loaded.filter_documents() == store.filter_documents()
        q = docs[3].embedding
        filters = {"field": "meta.year", "operator": ">=", "value": 2020}
        assert _ids(loaded.embedding_retrieval(q, filters=filters, roles=["staff"])) == _ids(
            store.embedding_retrieval(q, filters=filters, roles=["staff"])
        )
        loaded.write_documents([_doc("new", "n1", year=2030)])
        assert "n1" in _ids(loaded.filter_documents(filters))

    def test_empty_round_trip(self, store, tmp_path):
        store.save(tmp_path / "e")
        assert SimlarDocumentStore.load(tmp_path / "e").count_documents() == 0

    def test_misaligned_index_refused(self, corpus_store, tmp_path):
        store, _ = corpus_store
        store.save(tmp_path / "s")
        path = tmp_path / "s" / "store.json"
        data = json.loads(path.read_text())
        data["documents"] = data["documents"][::-1]
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="disagree"):
            SimlarDocumentStore.load(tmp_path / "s")

    def test_loads_legacy_layout(self, tmp_path):
        # simlar <= 1.0 layout: StreamingHelix index/ + tombstoned positions.
        docs = [_doc("alpha", "a", lang="en"), _doc("beta", "b", lang="fr"), _doc("gamma", "c")]
        root = tmp_path / "legacy"
        (root / "index").mkdir(parents=True)
        (root / "store.json").write_text(
            json.dumps(
                {
                    "corpus": [d.content for d in docs],
                    "documents": [d.to_dict() for d in docs],
                    "deleted_positions": [0],
                    "doc_id_to_pos": {"b": 1, "c": 2},
                    "init_parameters": {
                        "top_k": 3,
                        "relevance_k": 20,
                        "core_k": 10,
                        "parallel": False,
                    },
                }
            )
        )
        loaded = SimlarDocumentStore.load(root)
        assert loaded.to_dict()["init_parameters"]["top_k"] == 3
        assert _ids(loaded.filter_documents()) == ["b", "c"]
        assert _ids(
            loaded.filter_documents({"field": "meta.lang", "operator": "==", "value": "fr"})
        ) == ["b"]

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No saved store"):
            SimlarDocumentStore.load(tmp_path / "nonexistent")
