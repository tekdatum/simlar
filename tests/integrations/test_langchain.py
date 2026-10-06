"""Tests for the LangChain SimlarVectorStore.

The first group runs everywhere (engine stubs or the real engine). Filtering,
roles and ranking need the real engine: run with ``SIMLAR_REAL_ENGINE=1``.
"""

from __future__ import annotations

import asyncio
import pickle
import sys
import zlib

import numpy as np
import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from simlar.indexes.helix_index import HelixIndex
from simlar.integrations.langchain.simlar_vector_store import SimlarVectorStore

REAL_ENGINE = getattr(sys.modules.get("simlar_engine"), "__file__", None) is not None
real_engine = pytest.mark.skipif(
    not REAL_ENGINE, reason="needs the real engine (SIMLAR_REAL_ENGINE=1)"
)

DIM = 16


class _HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: shared words -> nearby vectors."""

    def _embed(self, text: str) -> list[float]:
        v = np.full(DIM, 0.01, dtype=np.float32)
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % DIM] += 1.0
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_CORPUS = [
    ("apple pie recipe baking", {"topic": "food", "year": 2019, "tags": ["dessert", "baking"]}),
    ("sourdough bread baking", {"topic": "food", "year": 2021, "tags": ["baking"]}),
    ("car engine repair", {"topic": "cars", "year": 2020, "tags": ["repair"]}),
    ("electric car battery", {"topic": "cars", "year": 2023, "tags": ["ev"]}),
    ("neural network training", {"topic": "ml", "year": 2022, "tags": ["ml", "dl"]}),
    ("gradient boosting trees", {"topic": "ml", "year": 2018, "tags": ["ml"]}),
]
_IDS = [f"d{i}" for i in range(len(_CORPUS))]
_ROLES = [["public"], ["public"], ["staff"], ["staff"], ["public", "staff"], ["staff"]]


def _store(**kw) -> SimlarVectorStore:
    return SimlarVectorStore.from_texts(
        [t for t, _ in _CORPUS],
        _HashEmbeddings(),
        metadatas=[dict(m) for _, m in _CORPUS],
        ids=list(_IDS),
        roles=_ROLES,
        parallel=False,
        **kw,
    )


def _ids(docs) -> list[str]:
    return [d.id for d in docs]


@pytest.fixture()
def empty():
    return SimlarVectorStore(embedding=_HashEmbeddings(), parallel=False)


# ── Basics (stubs or real engine) ────────────────────────────────────────────


class TestBasics:
    def test_empty_search_returns_empty(self, empty):
        assert empty.similarity_search("query", k=5) == []
        assert empty.similarity_search_by_vector([0.0] * DIM, k=5) == []

    def test_add_texts_returns_ids(self, empty):
        ids = empty.add_texts(["hello world", "foo bar"])
        assert len(ids) == 2 and len(set(ids)) == 2

    def test_add_texts_with_explicit_ids(self, empty):
        assert empty.add_texts(["cancer treatment", "ml transformers"], ids=["a", "b"]) == [
            "a",
            "b",
        ]

    def test_add_documents_uses_document_id(self, empty):
        ids = empty.add_documents(
            [Document(page_content="hello world", id="x"), Document(page_content="foo bar")]
        )
        assert ids[0] == "x" and ids[1]

    def test_add_empty_list_returns_empty(self, empty):
        assert empty.add_texts([]) == []

    def test_length_mismatches(self, empty):
        with pytest.raises(ValueError, match="ids length"):
            empty.add_documents([Document(page_content="a b")], ids=["a", "b"])
        with pytest.raises(ValueError, match="roles length"):
            empty.add_documents([Document(page_content="a b")], roles=[["x"], ["y"]])

    def test_similarity_search_returns_documents_with_metadata(self):
        docs = _store().similarity_search("baking", k=2)
        assert docs and all(isinstance(d, Document) for d in docs)
        assert all(d.metadata == dict(_CORPUS[int(d.id[1:])][1]) for d in docs)

    def test_similarity_search_with_score(self):
        results = _store().similarity_search_with_score("baking", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document) and isinstance(score, float)

    def test_get_by_ids(self):
        store = _store()
        docs = store.get_by_ids(["d3", "missing", "d0"])
        assert _ids(docs) == ["d3", "d0"]
        assert docs[0].page_content == "electric car battery"

    def test_returned_metadata_is_a_copy(self):
        store = _store()
        store.get_by_ids(["d0"])[0].metadata["topic"] = "changed"
        assert store.get_by_ids(["d0"])[0].metadata["topic"] == "food"

    def test_delete(self):
        store = _store()
        assert store.delete(["d1", "missing"]) is True
        assert store.delete(["missing"]) is False
        assert store.get_by_ids(["d1"]) == []
        assert store.index.ids == [i for i in _IDS if i != "d1"]

    def test_delete_all_then_add_again(self):
        store = _store()
        assert store.delete() is True
        assert store.similarity_search("baking") == []
        assert store.delete() is False
        store.add_texts(["fresh start text"], ids=["n"])
        assert _ids(store.similarity_search("fresh", k=1)) == ["n"]

    def test_add_only_indexes_the_new_batch(self, monkeypatch):
        """No rebuild: every add hands the index just the new documents."""
        seen: list[list[str]] = []
        orig = HelixIndex.add

        def spy(self, ids, *args, **kwargs):
            seen.append(list(ids))
            return orig(self, ids, *args, **kwargs)

        monkeypatch.setattr(HelixIndex, "add", spy)
        store = SimlarVectorStore(embedding=_HashEmbeddings(), parallel=False)
        for i in range(4):
            store.add_texts([f"batch {i} alpha", f"batch {i} beta"], ids=[f"{i}a", f"{i}b"])
        assert seen == [[f"{i}a", f"{i}b"] for i in range(4)]
        assert store.index.ids == [f"{i}{s}" for i in range(4) for s in "ab"]

    def test_re_add_updates_in_place(self, monkeypatch):
        added: list[list[str]] = []
        orig = HelixIndex.add

        def spy(self, ids, *args, **kwargs):
            added.append(list(ids))
            return orig(self, ids, *args, **kwargs)

        store = _store()
        monkeypatch.setattr(HelixIndex, "add", spy)
        store.add_texts(
            ["car engine repair manual", "new text here"],
            [{"topic": "cars"}, {}],
            ids=["d2", "new"],
        )
        assert added == [["new"]]  # d2 was updated, not re-added
        assert store.index.ids == [*_IDS, "new"]
        assert store.get_by_ids(["d2"])[0].page_content == "car engine repair manual"

    def test_from_texts_factory(self):
        s = SimlarVectorStore.from_texts(
            ["alpha text", "beta text", "gamma text"], embedding=_HashEmbeddings()
        )
        assert len(s.get_by_ids(s.index.ids)) == 3

    def test_embeddings_property(self, empty):
        assert isinstance(empty.embeddings, _HashEmbeddings)

    def test_save_and_load_roundtrip(self, tmp_path):
        store = _store()
        store.save_local(str(tmp_path))
        loaded = SimlarVectorStore.load_local(str(tmp_path), embedding=_HashEmbeddings())
        assert loaded.index.ids == _IDS
        assert loaded.get_by_ids(_IDS) == store.get_by_ids(_IDS)

    def test_save_and_load_empty(self, tmp_path, empty):
        empty.save_local(str(tmp_path))
        loaded = SimlarVectorStore.load_local(str(tmp_path), embedding=_HashEmbeddings())
        assert loaded.similarity_search("anything") == []

    def test_save_rejects_non_json_metadata(self, tmp_path, empty):
        empty.add_texts(["some text"], [{"obj": object()}])
        with pytest.raises(TypeError, match="JSON-serializable"):
            empty.save_local(str(tmp_path))


# ── Filtering, roles and ranking (real engine) ───────────────────────────────


@real_engine
class TestFiltering:
    def test_sql_filter(self):
        docs = _store().similarity_search("baking car", k=6, filter="year >= 2021")
        assert docs and set(_ids(docs)) <= {"d1", "d3", "d4"}

    def test_dict_filter(self):
        docs = _store().similarity_search(
            "car", k=6, filter={"topic": "cars", "year": {"$lt": 2023}}
        )
        assert _ids(docs) == ["d2"]

    def test_metadata_filter_chain(self):
        from simlar import MetadataFilter

        docs = _store().similarity_search(
            "training", k=6, filter=MetadataFilter().where_any("tags", "=", "ml")
        )
        assert set(_ids(docs)) == {"d4", "d5"}

    def test_multi_valued_filter(self):
        docs = _store().similarity_search("baking", k=6, filter="'baking' IN tags")
        assert set(_ids(docs)) == {"d0", "d1"}

    def test_filter_matching_nothing(self):
        assert _store().similarity_search("baking", k=3, filter="year > 3000") == []

    def test_unknown_column_raises(self):
        from simlar import FilterError

        with pytest.raises(FilterError):
            _store().similarity_search("baking", filter="nope = 1")

    def test_unfilterable_metadata_kept_but_not_queryable(self, empty):
        from simlar import FilterError

        empty.add_texts(
            ["nested metadata doc"], [{"info": {"a": 1}, "file-name": "x.txt", "ok": 1}], ids=["n"]
        )
        assert empty.get_by_ids(["n"])[0].metadata == {
            "info": {"a": 1},
            "file-name": "x.txt",
            "ok": 1,
        }
        assert _ids(empty.similarity_search("nested", filter="ok = 1")) == ["n"]
        with pytest.raises(FilterError):
            empty.similarity_search("nested", filter="info = 1")

    def test_unsafe_metadata_keys_are_not_indexed(self, empty):
        # "position" used to overwrite the SQL row's position, SQL keywords
        # broke ALTER TABLE, and ints beyond 64 bits overflowed SQLite.
        metas = [
            {"position": 7, "group": "g", "big": 2**70, "ok": 1},
            {"position": 0, "order": 3, "ok": 2},
            {"ok": 3},
        ]
        empty.add_texts(["first doc", "second doc", "third doc"], metas, ids=["a", "b", "c"])
        assert empty.index.sql_filter.ids == ["a", "b", "c"]
        assert set(_ids(empty.similarity_search("doc", k=3, filter="ok >= 2"))) == {"b", "c"}
        assert empty.get_by_ids(["a"])[0].metadata == metas[0]

    def test_roles(self):
        store = _store()
        assert set(_ids(store.similarity_search("car baking", k=6, roles=["public"]))) == {
            "d0",
            "d1",
            "d4",
        }
        assert store.similarity_search("car", k=6, roles=[]) == []
        store.grant(["d2"], ["public"])
        assert "d2" in _ids(store.similarity_search("car", k=6, roles=["public"]))
        store.revoke(["d2"], ["public"])
        assert "d2" not in _ids(store.similarity_search("car", k=6, roles=["public"]))

    def test_filter_and_roles_combined(self):
        docs = _store().similarity_search(
            "training trees", k=6, filter={"topic": "ml"}, roles=["public"]
        )
        assert _ids(docs) == ["d4"]

    def test_by_vector_with_filter(self):
        store = _store()
        q = _HashEmbeddings().embed_query("electric car battery")
        assert _ids(store.similarity_search_by_vector(q, k=1)) == ["d3"]
        assert _ids(store.similarity_search_by_vector(q, k=1, filter="topic = 'ml'"))[0] in {
            "d4",
            "d5",
        }

    def test_k_larger_than_store(self):
        assert len(_store().similarity_search("baking", k=100)) <= len(_CORPUS)

    def test_upsert_replaces_metadata(self):
        store = _store()
        store.add_texts(["apple pie recipe baking"], [{"topic": "archive"}], ids=["d0"])
        assert "d0" not in _ids(store.similarity_search("baking", k=6, filter="topic = 'food'"))
        assert "d0" in _ids(store.similarity_search("baking", k=6, filter="topic = 'archive'"))
        # Keys the new metadata dropped are cleared, not left behind.
        assert "d0" not in _ids(store.similarity_search("baking", k=6, filter="year = 2019"))
        assert "d0" not in _ids(store.similarity_search("baking", k=6, filter="'dessert' IN tags"))

    def test_upsert_replaces_text(self):
        store = _store()
        store.add_texts(["zebra migration savanna"], [{"topic": "animals"}], ids=["d3"])
        assert _ids(store.similarity_search("zebra savanna", k=1)) == ["d3"]

    def test_delete_keeps_filter_aligned(self):
        store = _store()
        store.similarity_search("car", filter="topic = 'cars'")  # populate the filter cache
        store.delete(["d0", "d2"])
        assert _ids(store.similarity_search("car", k=6, filter="topic = 'cars'")) == ["d3"]
        assert set(_ids(store.similarity_search("baking", k=6, roles=["public"]))) == {"d1", "d4"}
        store.add_texts(["new car review"], [{"topic": "cars"}], ids=["n"])
        assert set(_ids(store.similarity_search("car", k=6, filter="topic = 'cars'"))) == {
            "d3",
            "n",
        }

    def test_as_retriever_passes_filter_and_roles(self):
        retriever = _store().as_retriever(
            search_kwargs={"k": 6, "filter": "topic = 'ml'", "roles": ["public"]}
        )
        assert _ids(retriever.invoke("training")) == ["d4"]

    def test_relevance_scores_in_unit_range(self):
        results = _store().similarity_search_with_relevance_scores("baking bread", k=6)
        assert results and all(0.0 <= s <= 1.0 for _, s in results)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_async_search_runs_in_executor_threads(self):
        store = _store()

        async def run():
            return await asyncio.gather(
                *(store.asimilarity_search("car", k=6, filter="topic = 'cars'") for _ in range(8))
            )

        for docs in asyncio.run(run()):
            assert set(_ids(docs)) == {"d2", "d3"}

    def test_save_load_keeps_filter_and_roles(self, tmp_path):
        store = _store()
        q = dict(k=6, filter="year >= 2020", roles=["staff"])
        expected = _ids(store.similarity_search("car training", **q))
        store.save_local(str(tmp_path))
        loaded = SimlarVectorStore.load_local(str(tmp_path), embedding=_HashEmbeddings())
        assert _ids(loaded.similarity_search("car training", **q)) == expected

    def test_load_legacy_layout(self, tmp_path):
        """simlar 1.0 stores: HelixIndex in simlar/ + pickled sidecar."""
        texts = [t for t, _ in _CORPUS]
        metas = [dict(m) for _, m in _CORPUS]
        vectors = _HashEmbeddings().embed_documents(texts)
        helix = HelixIndex(text_k=500, vector_k=200, top_k=100)
        helix.add(ids=list(_IDS), texts=texts, vectors=np.array(vectors, dtype=np.float32))
        helix.save(str(tmp_path / "simlar"))
        sidecar = {
            "ids": list(_IDS),
            "texts": texts,
            "metadatas": metas,
            "vectors": vectors,
            "text_k": 500,
            "vector_k": 200,
            "top_k": 100,
            "parallel": False,
        }
        (tmp_path / "sidecar.pkl").write_bytes(pickle.dumps(sidecar))

        with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
            SimlarVectorStore.load_local(str(tmp_path), embedding=_HashEmbeddings())
        store = SimlarVectorStore.load_local(
            str(tmp_path), embedding=_HashEmbeddings(), allow_dangerous_deserialization=True
        )
        assert _ids(store.similarity_search("car", k=6, filter="topic = 'cars'")) in (
            ["d2", "d3"],
            ["d3", "d2"],
        )
        store.save_local(str(tmp_path / "converted"))
        assert (tmp_path / "converted" / "docs.json").exists()
