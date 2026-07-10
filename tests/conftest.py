"""
Inject a minimal simlar_engine stub into sys.modules before any simlar import.

Runs via pytest_configure (before collection) so test files can import simlar
at module level without the proprietary engine installed.

Covers every name imported from simlar_engine at module level across src/simlar/.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Stub dataclasses (mirrors private simlar_engine._types) ──────────────────


@dataclass
class _SearchResult:
    rank: int
    id: str
    score: float
    text: str | None = None


@dataclass
class _Parameters:
    boundaries: np.ndarray | None = None
    fit_values: np.ndarray | None = None


# ── Stub persistence (mirrors private simlar_engine._persistence) ─────────────

_FORMAT_VERSION = "1.0"


def _write_config(path, data: dict) -> None:
    payload = {"format_version": _FORMAT_VERSION, **data}
    Path(path).write_text(json.dumps(payload, indent=2))


def _read_config(path) -> dict:
    return json.loads(Path(path).read_text())


# ── Stub RRF (pure-Python path only) ─────────────────────────────────────────


class _ReciprocalRankFusion:
    def __init__(self, k: int = 2, weights=None):
        self._k = k
        self._weights = weights

    def __call__(self, results, k):
        if not results:
            return []
        weights = self._weights if self._weights is not None else [1.0] * len(results)
        id_to_text = {r.id: r.text for rs in results for r in rs if r.text is not None}
        scores: dict = {}
        for w, rs in zip(weights, results, strict=False):
            for r in rs:
                scores[r.id] = scores.get(r.id, 0.0) + w / (self._k + r.rank + 1)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            _SearchResult(rank=i, id=id_, score=score, text=id_to_text.get(id_))
            for i, (id_, score) in enumerate(top)
        ]


# ── Stub index cores ──────────────────────────────────────────────────────────


class _SimlarCore:
    def __init__(self, n_candidates=5000):
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._trained = False
        self._params: _Parameters | None = None
        self.coreindex = None
        self._matrix: np.ndarray | None = None

    def fit(self, embeddings, parallel=False, params=None, **kwargs):
        self._trained = True
        self._params = params

    def add(self, ids, vectors):
        self._ids = list(ids)
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._trained = True

    def update(self, ids, vectors):
        pass

    def delete(self, ids):
        pass

    def update_vector(self, doc_id: int, vector):
        pass

    def search(self, query, k=10):
        n = min(k, len(self._ids))
        return [_SearchResult(rank=i, id=self._ids[i], score=1.0 / (i + 1)) for i in range(n)]

    def search_raw(self, vectors, k, candidates=None, parallel=False, n_candidates=None):
        n = min(k, len(self._ids))
        return np.arange(n, dtype=np.int64), np.ones(n, dtype=np.float32)

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "simlar"})

    @classmethod
    def load(cls, directory):
        return cls()

    @property
    def size(self):
        return len(self._ids)

    @property
    def is_trained(self):
        return self._trained

    @property
    def index_type(self):
        return "simlar"


class _RelevanceCore:
    def __init__(self, method="robertson", k1=1.5, b=0.75, stopwords_lang=None, stemmer_lang=None):
        self._ids: list[str] = []
        self._trained = False

    def fit(self, texts, parallel=False, params=None, **kwargs):
        self._trained = True

    def add(self, ids, texts):
        self._ids = list(ids)
        self._trained = True

    def update(self, ids, texts):
        pass

    def delete(self, ids):
        pass

    def search(self, query, k=10):
        n = min(k, len(self._ids))
        return [_SearchResult(rank=i, id=self._ids[i], score=1.0 / (i + 1)) for i in range(n)]

    def search_raw(self, query, k, parallel=False):
        n = min(k, len(self._ids))
        return np.arange(n, dtype=np.int64), np.ones(n, dtype=np.float32)

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "bm25"})

    @classmethod
    def load(cls, directory):
        return cls()

    @property
    def size(self):
        return len(self._ids)

    @property
    def is_trained(self):
        return self._trained

    @property
    def index_type(self):
        return "bm25"


class _HelixCore:
    def __init__(
        self,
        text_index=None,
        vector_index=None,
        fusion=None,
        text_k=5000,
        vector_k=1000,
        top_k=100,
        alpha_text=0.10,
        alpha_vector=1.0,
    ):
        self._ids: list[str] = []
        self._trained = False
        self._text_index = text_index or _RelevanceCore()
        self._vector_index = vector_index or _SimlarCore()
        self._fusion = fusion or _ReciprocalRankFusion()
        self._text_k = text_k
        self._vector_k = vector_k
        self._top_k = top_k

    def fit(self, corpus, vectors, parallel=False, params=None):
        self._trained = True

    def add(self, ids, texts=None, vectors=None):
        self._ids = list(ids)
        self._trained = True

    def search(self, query_text=None, query_vector=None, k=None, parallel=False):
        effective_k = k or self._top_k
        n = min(effective_k, len(self._ids))
        return [_SearchResult(rank=i, id=self._ids[i], score=1.0 / (i + 1)) for i in range(n)]

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "helix"})

    @classmethod
    def load(cls, directory):
        return cls()

    @property
    def size(self):
        return len(self._ids)

    @property
    def is_trained(self):
        return self._trained

    @property
    def text_index(self):
        return self._text_index

    @property
    def vector_index(self):
        return self._vector_index

    @property
    def _params(self):
        return None

    @property
    def boundaries(self):
        return None

    @property
    def fit_values(self):
        return None


class _TextCore:
    def __init__(self, stopwords_lang="english", stemmer_lang="english"):
        self._ids: list[str] = []
        self._trained = False

    def fit(self, corpus, parallel=False, params=None, **kwargs):
        self._trained = True

    def add(self, ids, texts):
        self._ids = list(ids)
        self._trained = True

    def update(self, ids, texts):
        pass

    def delete(self, ids):
        self._ids = [i for i in self._ids if i not in set(ids)]

    def search(self, query, k=10):
        n = min(k, len(self._ids))
        return [_SearchResult(rank=i, id=self._ids[i], score=1.0 / (i + 1)) for i in range(n)]

    def search_raw(self, queries, k, parallel=False):
        n = min(k, len(self._ids))
        return np.arange(n, dtype=np.int64), np.ones(n, dtype=np.float32)

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "lookup"})

    @classmethod
    def load(cls, directory):
        return cls()

    @property
    def size(self):
        return len(self._ids)

    @property
    def is_trained(self):
        return self._trained

    @property
    def index_type(self):
        return "lookup"


class _StreamingCore:
    def __init__(self, **kwargs):
        self._count: int = 0
        self._trained = False

    def add_batch(self, corpus, vectors=None, parallel=False):
        self._count += len(corpus) if hasattr(corpus, "__len__") else 0
        self._trained = True

    def search(self, query_text=None, query_vector=None, k=10, parallel=False):
        n = min(k if k is not None else 10, self._count)
        return (
            np.arange(n, dtype=np.int64),
            np.array([1.0 / (i + 1) for i in range(n)], dtype=np.float32),
        )

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "streaming"})

    @classmethod
    def load(cls, directory):
        return cls()

    @property
    def size(self):
        return self._count

    @property
    def is_trained(self):
        return self._trained


# ── Inject stubs into sys.modules ─────────────────────────────────────────────


def _inject_engine_stubs() -> None:
    """Populate sys.modules with stub simlar_engine sub-modules."""
    if "simlar_engine" in sys.modules:
        return  # already installed (real engine or previously stubbed)

    def _mod(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    _mod(
        "simlar_engine",
        SearchResult=_SearchResult,
        _Parameters=_Parameters,
        FORMAT_VERSION=_FORMAT_VERSION,
        write_config=_write_config,
        read_config=_read_config,
        ReciprocalRankFusion=_ReciprocalRankFusion,
    )
    _mod("simlar_engine._types", SearchResult=_SearchResult, _Parameters=_Parameters)
    _mod(
        "simlar_engine._persistence",
        FORMAT_VERSION=_FORMAT_VERSION,
        write_config=_write_config,
        read_config=_read_config,
    )
    _mod("simlar_engine._registry")
    _mod("simlar_engine.fusion")
    _mod("simlar_engine.fusion.rrf", ReciprocalRankFusion=_ReciprocalRankFusion)
    _mod("simlar_engine.indexes")
    _mod("simlar_engine.indexes._simlar_impl", _SimlarCore=_SimlarCore)
    _mod("simlar_engine.indexes._helix_impl", _HelixCore=_HelixCore)
    _mod("simlar_engine.indexes._relevance_impl", _RelevanceCore=_RelevanceCore)
    _mod("simlar_engine.indexes._lookup_impl", _TextCore=_TextCore)
    _mod("simlar_engine.indexes._streaming_impl", _StreamingCore=_StreamingCore)
    _mod("simlar_engine.kernels")
    _mod("simlar_engine.kernels.fusion")


def pytest_configure(config) -> None:
    """Inject engine stubs before any test module is imported."""
    _inject_engine_stubs()


# ── Shared test helpers ───────────────────────────────────────────────────────


def make_results(ids: list[str], base_score: float = 1.0):
    from simlar.contracts import SearchResult

    return [SearchResult(rank=i, id=id_, score=base_score / (i + 1)) for i, id_ in enumerate(ids)]
