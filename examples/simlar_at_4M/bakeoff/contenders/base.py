import contextlib
import ctypes
import gc
import io
import time
from abc import ABC, abstractmethod

import numpy as np
import psutil

PROCESS = psutil.Process()


@contextlib.contextmanager
def quiet():
    """Silence an engine's progress bars so the run log stays readable."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _return_free_pages() -> None:
    """Hand pages the process no longer uses back to the OS, on glibc.

    Freeing an index does not shrink the process: malloc keeps the pages on its own free list, so
    the next engine allocates into that hole instead of growing. Its footprint would then measure
    as near-zero purely for having run after something large. `malloc_trim` releases those pages
    for real, which is what makes one engine's measurement independent of the ones before it.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):       # not glibc; deltas just get noisier
        pass


def resident_mb() -> float:
    """Physical memory the process holds, in MB, after letting go of all it can.

    Resident set size, not a Python allocation count: most of this field indexes in Rust or C++
    (simlar, TurboVec and Qdrant in Rust; FAISS and Milvus in C++), and `tracemalloc` cannot see
    a byte of it. RSS is the only measure that counts every engine in the same units.
    """
    gc.collect()
    _return_free_pages()
    return PROCESS.memory_info().rss / (1024 * 1024)


class Contender(ABC):
    """One retriever, built and timed exactly like every other one.

    A subclass says *what* happens in `_build` and `_rank`; this base class owns the stopwatch,
    so `build_ms` and `ms_per_query` mean the same thing for simlar and for the competition.
    Everyone indexes the same passages with the same embeddings.
    """

    name: str = "base"
    family: str = "competitor"

    def __init__(self, index_ids: list[str], texts: list[str], matrix: np.ndarray) -> None:
        self.index_ids = index_ids
        self.texts = texts
        self.matrix = matrix
        self.build_ms = float("nan")
        self.memory_mb = float("nan")
        self._search_ms = 0.0
        self._queries_served = 0

    @property
    def size(self) -> int:
        return len(self.index_ids)

    @property
    def ms_per_query(self) -> float:
        if not self._queries_served:
            return float("nan")
        return self._search_ms / self._queries_served

    def ids_at(self, positions) -> list[str]:
        """Map an engine's returned positions back to passage ids."""
        return [self.index_ids[int(position)] for position in positions]

    def warm_up(self) -> float:
        """Make the engine's first search happen off the clock, returning the cost in ms.

        Some engines defer expensive setup to their first call costing far more once than they
        ever will again. That is not the work being measured, and left alone it lands on whichever
        engine runs first, so the timings would depend on the order of CONTENDERS. Called after
        `build`, so the real index is ready.
        """
        with quiet():
            start = time.perf_counter()
            self._warm_up()
            total_ms = (time.perf_counter() - start) * 1000
        if total_ms >= 1:       # stays silent for the engines that had nothing to do
            print(f"  warmed {self.name:32s} {total_ms:8.1f} ms", flush=True)
        return total_ms

    def footprint_mb(self) -> float:
        """Memory currently held by wherever this engine keeps its index, in MB.

        Overridden by engines that do not keep it in this process: for those, the RSS of the
        benchmark says only how big their client stub is, which would report a server-backed
        index as costing almost nothing. Both readings of a delta use the same source, so
        whichever this returns, `memory_mb` measures the same thing before and after.
        """
        return resident_mb()

    def build(self) -> float:
        """Index the passages, returning how long it took in milliseconds.

        Also records `memory_mb`. Both readings settle the heap first, so what is measured is the
        index's steady state rather than the scratch space that building it happened to touch.
        They sit outside the stopwatch, so a slow reading cannot inflate `build_ms`.
        """
        before = self.footprint_mb()
        with quiet():
            start = time.perf_counter()
            self._build(self.texts, self.matrix)
            self.build_ms = (time.perf_counter() - start) * 1000
        self.memory_mb = self.footprint_mb() - before
        return self.build_ms

    def search(self, query_texts: list[str], query_matrix: np.ndarray, k: int) -> list[list[str]]:
        """Timed ranking: per query, the top-k passage ids, best first."""
        # with quiet():
        start = time.perf_counter()
        ranked = self._rank(query_texts, query_matrix, k)
        self._search_ms += (time.perf_counter() - start) * 1000
        self._queries_served += len(query_texts)
        return ranked

    @abstractmethod
    def _build(self, texts: list[str], matrix: np.ndarray) -> None:
        """Build the index over `texts` / `matrix`."""

    @abstractmethod
    def _warm_up(self) -> None:
        """Run one throwaway search, mirroring `_rank` with a single query.

        Required of every engine, even the ones with nothing to defer: what an engine leaves
        until its first call is not something you can tell by reading it, so it is measured
        rather than assumed. The result is discarded -- only the side effect matters.
        """

    @abstractmethod
    def _rank(self, query_texts: list[str], query_matrix: np.ndarray, k: int) -> list[list[str]]:
        """Return, per query, the top-k passage ids."""
