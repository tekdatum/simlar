from __future__ import annotations

import numpy as np
from simlar_engine.indexes._streaming_impl import _StreamingCore

from simlar.contracts import TextIndex, VectorIndex


class StreamingHelixIndex:
    """
    Example:
        >>> import numpy as np
        >>> idx = StreamingHelixIndex()
        >>> corpus = ["hello world", "foo bar"]
        >>> vecs = np.random.rand(2, 128).astype(np.float32)
        >>> idx.add_batch(corpus, vecs)
        >>> q_vec = np.random.rand(128).astype(np.float32)
        >>> ids, scores = idx.search("hello", q_vec, k=5)
    """

    def __init__(
        self,
        text_index_cls: type[TextIndex] | None = None,
        vector_index_cls: type[VectorIndex] | None = None,
        text_k: int | None = None,
        vector_k: int | None = None,
        top_k: int = 100,
        alpha_text: float = 0.10,
        alpha_vector: float = 1.0,
        rrf_k: int = 2,
        n_candidates: int | None = None,
    ) -> None:
        self._core = _StreamingCore(
            text_index_cls=text_index_cls,
            vector_index_cls=vector_index_cls,
            text_k=text_k,
            vector_k=vector_k,
            top_k=top_k,
            alpha_text=alpha_text,
            alpha_vector=alpha_vector,
            rrf_k=rrf_k,
            n_candidates=n_candidates,
        )

    # ── Ingest ────────────────────────────────────────────────────────────────

    def add_batch(
        self,
        corpus: list[str],
        vectors: np.ndarray,
        parallel: bool = False,
    ) -> None:
        self._core.add_batch(corpus, vectors, parallel)

    async def add_batch_async(
        self,
        corpus: list[str],
        vectors: np.ndarray,
        parallel: bool = False,
    ) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._core.add_batch, corpus, vectors, parallel)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str | list[str],
        query_vector: np.ndarray,
        k: int | None = None,
        parallel: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._core.search(query_text, query_vector, k, parallel)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        self._core.save(directory)

    @classmethod
    def load(cls, directory: str) -> StreamingHelixIndex:
        obj = cls.__new__(cls)
        obj._core = _StreamingCore.load(directory)
        return obj
