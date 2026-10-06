"""Tests for the LlamaIndex SimlarVectorStore.

Filter translation and basic behaviour run everywhere; filtering, ranking and
the VectorStoreIndex pipeline need the real engine (``SIMLAR_REAL_ENGINE=1``).
"""

from __future__ import annotations

import asyncio
import sys
import zlib

import numpy as np
import pytest

pytest.importorskip("llama_index.core", reason="llama-index-core not installed")

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from simlar.indexes.helix_index import HelixIndex
from simlar.integrations.llama_index.simlar_vector_store import SimlarVectorStore, to_engine_filter

REAL_ENGINE = getattr(sys.modules.get("simlar_engine"), "__file__", None) is not None
real_engine = pytest.mark.skipif(
    not REAL_ENGINE, reason="needs the real engine (SIMLAR_REAL_ENGINE=1)"
)

DIM = 16


def _embed(text: str) -> list[float]:
    v = np.full(DIM, 0.01, dtype=np.float32)
    for word in text.lower().split():
        v[zlib.crc32(word.encode()) % DIM] += 1.0
    return (v / np.linalg.norm(v)).tolist()


class _HashEmbedding(BaseEmbedding):
    """Deterministic bag-of-words embeddings: shared words -> nearby vectors."""

    def _get_query_embedding(self, query: str) -> list[float]:
        return _embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return _embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return _embed(text)


_CORPUS = [
    (
        "apple pie recipe baking",
        {"topic": "food", "year": 2019, "tags": ["dessert", "baking"], "author": "Ann Lee"},
    ),
    (
        "sourdough bread baking",
        {"topic": "food", "year": 2021, "tags": ["baking"], "author": "bob stone"},
    ),
    ("car engine repair", {"topic": "cars", "year": 2020, "tags": ["repair"]}),
    ("electric car battery", {"topic": "cars", "year": 2023, "tags": ["ev"]}),
    ("neural network training", {"topic": "ml", "year": 2022, "tags": ["ml", "dl"]}),
    ("gradient boosting trees", {"topic": "ml", "year": 2018, "tags": ["ml"]}),
]
_DOC_IDS = [f"doc{i}" for i in range(len(_CORPUS))]


def _index(store: SimlarVectorStore | None = None) -> VectorStoreIndex:
    store = store or SimlarVectorStore(parallel=False)
    docs = [
        Document(text=t, metadata=dict(m), id_=d)
        for (t, m), d in zip(_CORPUS, _DOC_IDS, strict=True)
    ]
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=_HashEmbedding(),
    )


def _refs(nodes) -> list[str]:
    return [n.node.ref_doc_id for n in nodes]


def _retrieve(index, query, *, filters=None, mode="default", k=6, **store_kwargs):
    retriever = index.as_retriever(
        similarity_top_k=k,
        filters=filters,
        vector_store_query_mode=mode,
        vector_store_kwargs=store_kwargs,
    )
    return retriever.retrieve(query)


def _f(key, value, op="=="):
    return MetadataFilter(key=key, value=value, operator=op)


def _fs(*filters, condition="and"):
    return MetadataFilters(filters=list(filters), condition=condition)


# ── Filter translation (pure) ────────────────────────────────────────────────

_KNOWN = {"topic", "year", "tags", "author"}


class TestToEngineFilter:
    @pytest.mark.parametrize(
        ("op", "value", "expected"),
        [
            ("==", "food", {"topic": {"$eq": "food"}}),
            ("!=", "food", {"topic": {"$ne": "food"}}),
            (">", 1, {"topic": {"$gt": 1}}),
            (">=", 1, {"topic": {"$gte": 1}}),
            ("<", 1, {"topic": {"$lt": 1}}),
            ("<=", 1, {"topic": {"$lte": 1}}),
            ("in", ["a", "b"], {"topic": {"$in": ["a", "b"]}}),
            ("nin", ["a"], {"topic": {"$nin": ["a"]}}),
            ("contains", "a", {"topic": {"$contains": "a"}}),
            ("any", ["a", "b"], {"topic": {"$in": ["a", "b"]}}),
            (
                "all",
                ["a", "b"],
                {"$and": [{"topic": {"$contains": "a"}}, {"topic": {"$contains": "b"}}]},
            ),
            ("text_match", "ab", {"topic": {"$like": "%ab%"}}),
            ("text_match_insensitive", "ab", {"topic": {"$like": "%ab%"}}),
            ("is_empty", None, {"topic": {"$exists": False}}),
        ],
    )
    def test_operators(self, op, value, expected):
        assert to_engine_filter(_fs(_f("topic", value, op)), _KNOWN) == expected

    def test_conditions_and_nesting(self):
        a, b = _f("topic", "food"), _f("year", 2020, ">=")
        assert to_engine_filter(_fs(a, b), _KNOWN) == {
            "$and": [{"topic": {"$eq": "food"}}, {"year": {"$gte": 2020}}]
        }
        assert to_engine_filter(_fs(a, b, condition="or"), _KNOWN) == {
            "$or": [{"topic": {"$eq": "food"}}, {"year": {"$gte": 2020}}]
        }
        assert to_engine_filter(_fs(a, b, condition="not"), _KNOWN) == {
            "$not": {"$or": [{"topic": {"$eq": "food"}}, {"year": {"$gte": 2020}}]}
        }
        nested = _fs(_fs(a, b, condition="or"), _f("tags", "ml", "contains"))
        assert to_engine_filter(nested, _KNOWN) == {
            "$and": [
                {"$or": [{"topic": {"$eq": "food"}}, {"year": {"$gte": 2020}}]},
                {"tags": {"$contains": "ml"}},
            ]
        }

    def test_unknown_keys_match_nothing_except_is_empty(self):
        assert to_engine_filter(_fs(_f("nope", 1)), _KNOWN) == {"position": {"$lt": 0}}
        assert to_engine_filter(_fs(_f("nope", None, "is_empty")), _KNOWN) == {
            "position": {"$gte": 0}
        }

    def test_empty(self):
        assert to_engine_filter(None, _KNOWN) is None
        assert to_engine_filter(MetadataFilters(filters=[]), _KNOWN) is None


# ── Basics (stubs or real engine) ────────────────────────────────────────────


class TestBasics:
    def test_add_requires_embeddings(self):
        with pytest.raises(ValueError, match="embeddings"):
            SimlarVectorStore().add([TextNode(text="no embedding here")])

    def test_add_returns_node_ids(self):
        store = SimlarVectorStore(parallel=False)
        nodes = [
            TextNode(id_=f"n{i}", text=f"text number {i}", embedding=_embed(f"t{i}"))
            for i in range(3)
        ]
        assert store.add(nodes) == ["n0", "n1", "n2"]
        assert store.client.ids == ["n0", "n1", "n2"]

    def test_add_empty(self):
        assert SimlarVectorStore().add([]) == []

    def test_empty_store_query(self):
        result = SimlarVectorStore().query(
            VectorStoreQuery(query_embedding=_embed("x"), similarity_top_k=3)
        )
        assert result.nodes == [] and result.ids == []

    def test_unsupported_mode(self):
        with pytest.raises(NotImplementedError):
            SimlarVectorStore().query(
                VectorStoreQuery(query_embedding=_embed("x"), mode=VectorStoreQueryMode.MMR)
            )

    def test_mode_requirements(self):
        store = SimlarVectorStore()
        with pytest.raises(ValueError, match="query_embedding"):
            store.query(VectorStoreQuery(query_str="x"))
        with pytest.raises(ValueError, match="query_str"):
            store.query(
                VectorStoreQuery(query_embedding=_embed("x"), mode=VectorStoreQueryMode.SPARSE)
            )

    def test_add_only_indexes_the_new_batch(self, monkeypatch):
        seen: list[list[str]] = []
        orig = HelixIndex.add

        def spy(self, ids, *args, **kwargs):
            seen.append(list(ids))
            return orig(self, ids, *args, **kwargs)

        monkeypatch.setattr(HelixIndex, "add", spy)
        store = SimlarVectorStore(parallel=False)
        for b in range(3):
            store.add(
                [TextNode(id_=f"{b}{s}", text=f"batch {b} {s}", embedding=_embed(s)) for s in "ab"]
            )
        assert seen == [[f"{b}a", f"{b}b"] for b in range(3)]

    def test_clear(self):
        store = SimlarVectorStore(parallel=False)
        store.add([TextNode(id_="a", text="alpha text", embedding=_embed("alpha"))])
        store.clear()
        assert store.client.ids == []


# ── Pipeline, filtering and ranking (real engine) ────────────────────────────


@real_engine
class TestPipeline:
    def test_default_mode_is_vector_search(self):
        nodes = _retrieve(_index(), "electric car battery", k=1)
        assert _refs(nodes) == ["doc3"]

    def test_sparse_mode_is_text_search(self):
        nodes = _retrieve(_index(), "sourdough", mode="sparse", k=1)
        assert _refs(nodes) == ["doc1"]

    def test_hybrid_mode(self):
        nodes = _retrieve(_index(), "gradient boosting trees", mode="hybrid", k=2)
        assert _refs(nodes)[0] == "doc5"

    def test_node_round_trip(self):
        node = _retrieve(_index(), "apple pie", mode="sparse", k=1)[0].node
        assert node.ref_doc_id == "doc0"
        assert node.get_content() == "apple pie recipe baking"
        assert node.metadata == _CORPUS[0][1]

    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            (_fs(_f("topic", "cars")), {"doc2", "doc3"}),
            (_fs(_f("topic", "cars", "!=")), {"doc0", "doc1", "doc4", "doc5"}),
            (_fs(_f("year", 2021, ">=")), {"doc1", "doc3", "doc4"}),
            (_fs(_f("year", 2019, "<")), {"doc5"}),
            (_fs(_f("topic", ["ml", "cars"], "in")), {"doc2", "doc3", "doc4", "doc5"}),
            (_fs(_f("topic", ["ml", "cars"], "nin")), {"doc0", "doc1"}),
            (_fs(_f("tags", "ml", "contains")), {"doc4", "doc5"}),
            (_fs(_f("tags", ["ev", "dessert"], "any")), {"doc0", "doc3"}),
            (_fs(_f("tags", ["ml", "dl"], "all")), {"doc4"}),
            (_fs(_f("author", "stone", "text_match")), {"doc1"}),
            (_fs(_f("author", "ANN", "text_match_insensitive")), {"doc0"}),
            (_fs(_f("author", None, "is_empty")), {"doc2", "doc3", "doc4", "doc5"}),
            (_fs(_f("topic", "food"), _f("year", 2020, ">"), condition="and"), {"doc1"}),
            (_fs(_f("topic", "food"), _f("year", 2023), condition="or"), {"doc0", "doc1", "doc3"}),
            (_fs(_f("topic", "food"), _f("topic", "ml"), condition="not"), {"doc2", "doc3"}),
            (
                _fs(
                    _fs(_f("year", 2019, ">="), _f("year", 2021, "<="), condition="and"),
                    _fs(_f("topic", "food"), _f("topic", "cars"), condition="or"),
                ),
                {"doc0", "doc1", "doc2"},
            ),
            (_fs(_f("unknown_key", "x")), set()),
            (_fs(_f("unknown_key", None, "is_empty")), set(_DOC_IDS)),
        ],
    )
    def test_metadata_filters(self, filters, expected):
        assert set(_refs(_retrieve(_index(), "baking car training", filters=filters))) == expected

    def test_doc_ids_and_node_ids(self):
        index = _index()
        store = index.vector_store
        q = VectorStoreQuery(
            query_embedding=_embed("car"), similarity_top_k=6, doc_ids=["doc2", "doc4"]
        )
        assert {n.ref_doc_id for n in store.query(q).nodes} == {"doc2", "doc4"}
        node_id = store.query(q).ids[0]
        q = VectorStoreQuery(query_embedding=_embed("car"), similarity_top_k=6, node_ids=[node_id])
        assert store.query(q).ids == [node_id]

    def test_filter_and_roles(self):
        store = SimlarVectorStore(parallel=False)
        nodes = [
            TextNode(id_=f"n{i}", text=t, metadata=dict(m), embedding=_embed(t))
            for i, (t, m) in enumerate(_CORPUS)
        ]
        store.add(
            nodes,
            roles=[["public"], ["public"], ["staff"], ["staff"], ["public", "staff"], ["staff"]],
        )
        index = VectorStoreIndex.from_vector_store(store, embed_model=_HashEmbedding())
        got = _retrieve(index, "training", filters=_fs(_f("topic", "ml")), roles=["public"])
        assert [n.node.node_id for n in got] == ["n4"]
        store.grant(["n5"], ["public"])
        assert {
            n.node.node_id
            for n in _retrieve(index, "training", filters=_fs(_f("topic", "ml")), roles=["public"])
        } == {"n4", "n5"}
        store.revoke(["n5"], ["public"])
        assert _retrieve(index, "x", roles=[]) == []

    def test_delete_ref_doc(self):
        index = _index()
        index.delete_ref_doc("doc2")
        assert "doc2" not in _refs(_retrieve(index, "car engine repair"))
        assert set(_refs(_retrieve(index, "car", filters=_fs(_f("topic", "cars"))))) == {"doc3"}

    def test_delete_nodes_and_get_nodes(self):
        store = _index().vector_store
        ml = store.get_nodes(filters=_fs(_f("topic", "ml")))
        assert {n.ref_doc_id for n in ml} == {"doc4", "doc5"}
        assert [n.node_id for n in store.get_nodes(node_ids=[ml[0].node_id])] == [ml[0].node_id]
        store.delete_nodes(filters=_fs(_f("topic", "ml")))
        assert store.get_nodes(filters=_fs(_f("topic", "ml"))) == []
        store.delete_nodes()  # no-op
        assert len(store.client.ids) == 4
        with pytest.raises(ValueError):
            store.get_nodes()

    def test_delete_everything_then_add(self):
        index = _index()
        store = index.vector_store
        store.delete_nodes(node_ids=list(store.client.ids))
        assert store.client.ids == []
        index.insert(Document(text="fresh start text", id_="new"))
        assert _refs(_retrieve(index, "fresh start")) == ["new"]

    def test_upsert_replaces_metadata(self):
        store = SimlarVectorStore(parallel=False)
        store.add(
            [
                TextNode(
                    id_="a",
                    text="alpha beta",
                    metadata={"topic": "x", "year": 1},
                    embedding=_embed("a"),
                )
            ]
        )
        store.add(
            [TextNode(id_="b", text="gamma delta", metadata={"topic": "y"}, embedding=_embed("b"))]
        )
        store.add(
            [TextNode(id_="a", text="alpha beta", metadata={"topic": "z"}, embedding=_embed("a"))]
        )
        assert store.client.ids == ["a", "b"]
        assert [n.node_id for n in store.get_nodes(filters=_fs(_f("topic", "z")))] == ["a"]
        assert store.get_nodes(filters=_fs(_f("year", 1))) == []
        assert store.get_nodes(node_ids=["a"])[0].metadata == {"topic": "z"}

    def test_persist_round_trip(self, tmp_path):
        index = _index()
        index.storage_context.persist(persist_dir=str(tmp_path))
        store = SimlarVectorStore.from_persist_dir(str(tmp_path))
        loaded = VectorStoreIndex.from_vector_store(store, embed_model=_HashEmbedding())
        f = _fs(_f("year", 2020, ">="))
        assert _refs(_retrieve(loaded, "car", filters=f)) == _refs(
            _retrieve(index, "car", filters=f)
        )

    def test_persist_missing(self, tmp_path):
        with pytest.raises(ValueError, match="No saved"):
            SimlarVectorStore.from_persist_path(str(tmp_path / "nope"))

    def test_async_retrieve(self):
        index = _index()
        retriever = index.as_retriever(similarity_top_k=6, filters=_fs(_f("topic", "cars")))

        async def run():
            return await asyncio.gather(*(retriever.aretrieve("car") for _ in range(4)))

        for nodes in asyncio.run(run()):
            assert set(_refs(nodes)) == {"doc2", "doc3"}
