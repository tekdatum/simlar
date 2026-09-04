from __future__ import annotations

import logging
import pickle
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from simlar.contracts import SearchResult
from simlar.indexes.helix_index import HelixIndex

logger = logging.getLogger(__name__)

_INDEX_DIRNAME = "simlar"
_SIDECAR_FILENAME = "sidecar.pkl"


class SimlarVectorStore(VectorStore):
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
            top_k: Final result list length from the HelixIndex.
            parallel: Default threading mode for index writes and searches.
                Override per call by passing ``parallel=`` to ``add_texts`` or
                the ``similarity_search*`` methods.
        """
        self._embedding = embedding
        self._text_k = text_k
        self._vector_k = vector_k
        self._top_k = top_k
        self._parallel = parallel
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._vectors: list[list[float]] = []  # stored to avoid re-embedding on add
        self._id_to_pos: dict[str, int] = {}
        self._index = self._make_index()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_index(self) -> HelixIndex:
        return HelixIndex(
            text_k=self._text_k,
            vector_k=self._vector_k,
            top_k=self._top_k,
        )

    def _raw_search(
        self,
        query: str,
        k: int,
        parallel: bool | None = None,
    ) -> list[tuple[Document, float]]:
        if not self._texts:
            return []

        query_vector = np.array(self._embedding.embed_query(query), dtype=np.float32)
        effective_k = min(k, len(self._texts))
        results = cast(
            list[SearchResult],
            self._index.search(
                query_text=query,
                query_vector=query_vector,
                k=effective_k,
                parallel=self._parallel if parallel is None else parallel,
            ),
        )

        out = []
        for result in results:
            pos = self._id_to_pos.get(result.id)
            if pos is None:
                continue
            doc = Document(
                page_content=self._texts[pos],
                metadata={**self._metadatas[pos], "_simlar_score": result.score},
                id=result.id,
            )
            out.append((doc, result.score))
        return out

    # ── VectorStore contract ───────────────────────────────────────────────────

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts

        New embeddings are computed only for the incoming texts; previously
        stored embeddings are reused.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata dicts, one per text.
            ids: Optional explicit string IDs. Generated as UUIDs if omitted.
            parallel: Thread the index rebuild for this call. Defaults to the
                store-level setting.

        Returns:
            List of IDs for the added (or updated) documents.
        """
        parallel = kwargs.pop("parallel", self._parallel)
        texts_list = list(texts)
        if not texts_list:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        if len(ids) != len(texts_list):
            raise ValueError(f"ids length {len(ids)} != texts length {len(texts_list)}")
        if len(metadatas) != len(texts_list):
            raise ValueError(f"metadatas length {len(metadatas)} != texts length {len(texts_list)}")

        new_vectors = self._embedding.embed_documents(texts_list)

        for id_, text, meta, vec in zip(ids, texts_list, metadatas, new_vectors, strict=False):
            if id_ in self._id_to_pos:
                pos = self._id_to_pos[id_]
                self._texts[pos] = text
                self._metadatas[pos] = meta
                self._vectors[pos] = vec
            else:
                pos = len(self._ids)
                self._ids.append(id_)
                self._texts.append(text)
                self._metadatas.append(meta)
                self._vectors.append(vec)
                self._id_to_pos[id_] = pos

        self._index = self._make_index()
        self._index.add(
            ids=self._ids,
            texts=self._texts,
            vectors=np.array(self._vectors, dtype=np.float32),
            parallel=parallel,
        )

        return list(ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        return [doc for doc, _ in self._raw_search(query, k, kwargs.get("parallel"))]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return self._raw_search(query, k, kwargs.get("parallel"))

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
                )
        """
        store = cls(
            embedding=embedding,
            text_k=text_k,
            vector_k=vector_k,
            top_k=top_k,
            parallel=parallel,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_local(self, folder: str) -> None:
        """Save the index and document sidecar to disk.

        Layout::

            <folder>/
                simlar_index/   <- HelixIndex native format
                sidecar.pkl     <- texts, metadatas, ids, embeddings, k-chain params

        .. note::
            The same embedding model (or one with an identical output dimension)
            must be supplied on ``load_local`` — this is not validated at load time.
        """
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        self._index.save(str(d / _INDEX_DIRNAME))
        with open(d / _SIDECAR_FILENAME, "wb") as f:
            pickle.dump(
                {
                    "ids": self._ids,
                    "texts": self._texts,
                    "metadatas": self._metadatas,
                    "vectors": self._vectors,
                    "text_k": self._text_k,
                    "vector_k": self._vector_k,
                    "top_k": self._top_k,
                    "parallel": self._parallel,
                },
                f,
            )
        logger.info("SimlarVectorStore saved to %s (%d documents)", folder, len(self._ids))

    @classmethod
    def load_local(
        cls,
        folder: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> SimlarVectorStore:
        """Load a SimlarVectorStore saved with ``save_local``.

        Args:
            folder: Directory written by ``save_local``.
            embedding: Embeddings model for future queries. Must have the same
                output dimension as the model used when building the index.
                Documents are NOT re-embedded on load.
            parallel: Optional keyword overriding the threading mode saved in
                the sidecar.
        """
        d = Path(folder)
        with open(d / _SIDECAR_FILENAME, "rb") as f:
            sidecar = pickle.load(f)

        store = cls(
            embedding=embedding,
            text_k=sidecar["text_k"],
            vector_k=sidecar["vector_k"],
            top_k=sidecar["top_k"],
            parallel=kwargs.get("parallel", sidecar.get("parallel", True)),
        )
        store._ids = sidecar["ids"]
        store._texts = sidecar["texts"]
        store._metadatas = sidecar["metadatas"]
        store._vectors = sidecar["vectors"]
        store._id_to_pos = {id_: pos for pos, id_ in enumerate(store._ids)}
        store._index = HelixIndex.load(str(d / _INDEX_DIRNAME))

        logger.info("SimlarVectorStore loaded from %s (%d documents)", folder, len(store._ids))
        return store
