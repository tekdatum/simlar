"""LangChain ``VectorStore`` backed by simlar's FilteredIndex over a HelixIndex.

Hybrid (BM25 + vector) search with SQL metadata filtering and role-based
access. Every write goes straight to the index -- new ids are appended,
re-added ids are updated in place, deletes compact -- so nothing is ever
rebuilt and no copy of the embeddings is kept here (the engine holds its own
quantized one). The store only keeps each document's text and metadata, to
hand results back as ``Document`` objects.

Example::

    store = SimlarVectorStore.from_texts(texts, embedding, metadatas=metas)
    store.similarity_search("query", k=5, filter="year >= 2020 AND 'ml' IN tags")
    retriever = store.as_retriever(search_kwargs={"k": 5, "filter": {"lang": "en"}})
"""

from __future__ import annotations

import json
import logging
import pickle
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from simlar.indexes.filtered_index import FilteredIndex, MetadataFilter, SQLFilter
from simlar.indexes.helix_index import HelixIndex
from simlar.integrations._filtering import filterable_metadata, replacement_patch

logger = logging.getLogger(__name__)

_INDEX_DIRNAME = "index"
_DOCS_FILENAME = "docs.json"
# Layout written before filtering existed: HelixIndex + pickled sidecar.
_LEGACY_INDEX_DIRNAME = "simlar"
_LEGACY_SIDECAR_FILENAME = "sidecar.pkl"

Filter: TypeAlias = "str | dict | MetadataFilter"


class SimlarVectorStore(VectorStore):
    """Hybrid search with metadata filtering over a simlar ``FilteredIndex``.

    ``filter`` on every search method is a SQL WHERE fragment
    (``"len >= 6 AND lang IN ('en', 'es')"``, ``"'ml' IN tags"``), a
    LangChain-style dict (``{"len": {"$gte": 6}}``) or a ``MetadataFilter``
    chain. ``roles=[...]`` restricts results to documents granted to any of
    those roles; ``None`` (default) skips the access check.

    Metadata keys that are identifiers with scalar (or list-of-scalar) values
    are filterable; list values are multi-valued (``"'ml' IN tags"``). Other
    metadata is returned on the Document but can't be filtered on.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        text_k: int = 500,
        vector_k: int = 200,
        top_k: int = 100,
        parallel: bool = True,
    ) -> None:
        """
        Args:
            embedding: Embeddings model used for documents and queries.
            text_k: Text candidate pool size fed into RRF.
            vector_k: Vector candidate pool size fed into RRF.
            top_k: Default result list length of the HelixIndex.
            parallel: Default threading mode for index writes and searches.
                Override per call with ``parallel=``.
        """
        self._embedding = embedding
        self._text_k = text_k
        self._vector_k = vector_k
        self._top_k = top_k
        self._parallel = parallel
        self._docs: dict[str, tuple[str, dict]] = {}
        self._index = self._new_index()

    def _new_index(self) -> FilteredIndex:
        return FilteredIndex(
            HelixIndex(text_k=self._text_k, vector_k=self._vector_k, top_k=self._top_k)
        )

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def index(self) -> FilteredIndex:
        """The underlying ``FilteredIndex(HelixIndex)``."""
        return self._index

    # ── Writes ─────────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents, or replace the ones whose id is already stored.

        Only these documents are embedded and indexed: new ids are appended
        to the index, existing ids are updated in place (text, vector and
        metadata; keys the new metadata drops are cleared).

        Args:
            documents: Documents to add. ``Document.id`` is used as the id when
                set; otherwise a UUID is generated.
            ids: Optional explicit ids, taking precedence over ``Document.id``.
            roles: Optional list of role lists, one per document, granting
                those roles access for ``roles=`` searches (additive).
            parallel: Thread this write. Defaults to the store-level setting.

        Returns:
            The ids of the added or replaced documents.
        """
        ids = kwargs.pop("ids", None)
        roles = kwargs.pop("roles", None)
        parallel = kwargs.pop("parallel", None)
        n = len(documents)
        if ids is None:
            ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        elif len(ids) != n:
            raise ValueError(f"ids length {len(ids)} != documents length {n}")
        if roles is not None and len(roles) != n:
            raise ValueError(f"roles length {len(roles)} != documents length {n}")
        if n == 0:
            return []
        ids = list(ids)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        vectors = np.asarray(self._embedding.embed_documents(texts), dtype=np.float32)
        parallel = self._parallel if parallel is None else parallel

        # A repeated id keeps its last occurrence.
        last = {id_: i for i, id_ in enumerate(ids)}
        order = list(last.values())
        new = [i for i in order if ids[i] not in self._docs]
        old = [i for i in order if ids[i] in self._docs]

        if new:
            self._index.add(
                [ids[i] for i in new],
                [texts[i] for i in new],
                vectors if len(new) == n else vectors[new],
                metadata=[filterable_metadata(metadatas[i]) for i in new],
                roles=None if roles is None else [roles[i] for i in new],
                parallel=parallel,
            )
        if old:
            self._index.update(
                [ids[i] for i in old],
                [texts[i] for i in old],
                vectors if len(old) == n else vectors[old],
                metadata=[replacement_patch(self._docs[ids[i]][1], metadatas[i]) for i in old],
            )
            for i in old:
                if roles is not None and roles[i]:
                    self._index.grant([ids[i]], list(roles[i]))

        for i in order:
            self._docs[ids[i]] = (texts[i], metadatas[i])
        return ids

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by id; ``ids=None`` deletes everything.

        Unknown ids are ignored. Returns True if anything was deleted.
        """
        doomed = (
            list(self._docs) if ids is None else [i for i in dict.fromkeys(ids) if i in self._docs]
        )
        if not doomed:
            return False
        if len(doomed) == len(self._docs):
            # The BM25 core refuses to delete its last document; start fresh.
            self._docs.clear()
            self._index = self._new_index()
            return True
        self._index.delete(doomed)
        for id_ in doomed:
            del self._docs[id_]
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Documents for ``ids``, in the order given; unknown ids are skipped."""
        return [self._document(id_) for id_ in ids if id_ in self._docs]

    def grant(self, ids: list[str], roles: list[str]) -> None:
        """Let documents ``ids`` be returned to searches made with any of ``roles``."""
        self._index.grant(ids, roles)

    def revoke(self, ids: list[str], roles: list[str]) -> None:
        """Remove ``roles``' access to documents ``ids``."""
        self._index.revoke(ids, roles)

    # ── Search ─────────────────────────────────────────────────────────────────

    def _document(self, id_: str) -> Document:
        text, meta = self._docs[id_]
        return Document(page_content=text, metadata=dict(meta), id=id_)

    def _search(
        self,
        query: str | None,
        embedding: Any,
        k: int,
        filter: Filter | None,
        kwargs: dict[str, Any],
    ) -> list[tuple[Document, float]]:
        if not self._docs or k <= 0:
            return []
        if embedding is None:
            assert query is not None
            embedding = self._embedding.embed_query(query)
        parallel = kwargs.get("parallel")
        results = self._index.search(
            query_text=query,
            query_vector=np.asarray(embedding, dtype=np.float32),
            k=min(k, len(self._docs)),
            parallel=self._parallel if parallel is None else parallel,
            filter=filter,
            roles=kwargs.get("roles"),
        )
        return [(self._document(r.id), float(r.score)) for r in results]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Filter | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Hybrid (BM25 + vector) search.

        Args:
            query: Query text.
            k: Number of documents to return.
            filter: Restrict results by metadata before ranking (see class docs).
            roles: Only return documents granted to one of these roles.
            parallel: Override the store-level threading mode.
        """
        return [doc for doc, _ in self._search(query, None, k, filter, kwargs)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Filter | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Like ``similarity_search``, returning ``(Document, score)`` pairs;
        higher scores are more relevant."""
        return self._search(query, None, k, filter, kwargs)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Filter | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Vector-only search for an already-embedded query. Takes the same
        ``filter`` / ``roles`` / ``parallel`` options as ``similarity_search``."""
        return [doc for doc, _ in self._search(None, embedding, k, filter, kwargs)]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Filter | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Like ``similarity_search_by_vector``, returning ``(Document, score)`` pairs."""
        return self._search(None, embedding, k, filter, kwargs)

    def _select_relevance_score_fn(self):
        # Scores are already higher-is-better in [0, 1].
        return lambda score: score

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        text_k: int = 500,
        vector_k: int = 200,
        top_k: int = 100,
        parallel: bool = True,
        **kwargs: Any,
    ) -> SimlarVectorStore:
        """Build a SimlarVectorStore from a list of texts.

        Example:
            .. code-block:: python

                store = SimlarVectorStore.from_texts(
                    texts=["cancer treatment", "machine learning"],
                    embedding=OpenAIEmbeddings(),
                    metadatas=[{"source": "a"}, {"source": "b"}],
                    roles=[["public"], ["staff"]],
                )
        """
        store = cls(
            embedding=embedding,
            text_k=text_k,
            vector_k=vector_k,
            top_k=top_k,
            parallel=parallel,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids, roles=kwargs.get("roles"))
        return store

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_local(self, folder: str) -> None:
        """Save the store to a directory.

        Layout::

            <folder>/
                index/      <- FilteredIndex: HelixIndex + SQLFilter (metadata, roles)
                docs.json   <- texts, metadata and store settings

        Metadata must be JSON-serializable. The same embedding model (or one
        with the same output dimension) must be supplied to ``load_local``.
        """
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        if self._docs:
            self._index.save(str(d / _INDEX_DIRNAME))
        payload = {
            "text_k": self._text_k,
            "vector_k": self._vector_k,
            "top_k": self._top_k,
            "parallel": self._parallel,
            # In index order, so load_local can check alignment.
            "docs": [[id_, *self._docs[id_]] for id_ in (self._index.ids if self._docs else [])],
        }
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except TypeError as e:
            raise TypeError(f"SimlarVectorStore metadata must be JSON-serializable: {e}") from None
        (d / _DOCS_FILENAME).write_text(text, encoding="utf-8")
        logger.info("SimlarVectorStore saved to %s (%d documents)", folder, len(self._docs))

    @classmethod
    def load_local(
        cls,
        folder: str,
        embedding: Embeddings,
        *,
        allow_dangerous_deserialization: bool = False,
        **kwargs: Any,
    ) -> SimlarVectorStore:
        """Load a store saved with ``save_local``.

        Args:
            folder: Directory written by ``save_local``.
            embedding: Embeddings model for queries and future writes. Must have
                the same output dimension as the one the store was built with.
            allow_dangerous_deserialization: Required to read a store saved by
                simlar < 1.1 (``sidecar.pkl``), which is a pickle.
            parallel: Optional keyword overriding the saved threading mode.
        """
        d = Path(folder)
        if not (d / _DOCS_FILENAME).exists() and (d / _LEGACY_SIDECAR_FILENAME).exists():
            return cls._load_legacy(d, embedding, allow_dangerous_deserialization, kwargs)

        payload = json.loads((d / _DOCS_FILENAME).read_text(encoding="utf-8"))
        store = cls(
            embedding=embedding,
            text_k=payload["text_k"],
            vector_k=payload["vector_k"],
            top_k=payload["top_k"],
            parallel=kwargs.get("parallel", payload["parallel"]),
        )
        store._docs = {id_: (text, meta) for id_, text, meta in payload["docs"]}
        if store._docs:
            store._index = FilteredIndex.load(str(d / _INDEX_DIRNAME))
            if store._index.ids != list(store._docs):
                raise ValueError(f"Corrupted store at {folder!r}: index and documents disagree.")
        logger.info("SimlarVectorStore loaded from %s (%d documents)", folder, len(store._docs))
        return store

    @classmethod
    def _load_legacy(
        cls, d: Path, embedding: Embeddings, allow: bool, kwargs: dict[str, Any]
    ) -> SimlarVectorStore:
        if not allow:
            raise ValueError(
                f"{d} was saved by simlar < 1.1 as a pickle, which can run arbitrary code "
                f"when loaded. Pass allow_dangerous_deserialization=True if you trust it, "
                f"then save_local() again to convert it."
            )
        with open(d / _LEGACY_SIDECAR_FILENAME, "rb") as f:
            sidecar = pickle.load(f)  # noqa: S301 -- opt-in above
        store = cls(
            embedding=embedding,
            text_k=sidecar["text_k"],
            vector_k=sidecar["vector_k"],
            top_k=sidecar["top_k"],
            parallel=kwargs.get("parallel", sidecar.get("parallel", True)),
        )
        ids = sidecar["ids"]
        store._docs = {
            id_: (text, meta)
            for id_, text, meta in zip(ids, sidecar["texts"], sidecar["metadatas"], strict=True)
        }
        if ids:
            # The saved HelixIndex is reused as is; only the metadata filter is
            # new, built in the same id order the index was.
            sql_filter = SQLFilter()
            sql_filter.add(ids, [filterable_metadata(m) for _, m in store._docs.values()])
            store._index = FilteredIndex(
                HelixIndex.load(str(d / _LEGACY_INDEX_DIRNAME)), sql_filter=sql_filter
            )
        return store
