"""
Inject a minimal simlar_engine stub into sys.modules before any simlar import.

Runs via pytest_configure (before collection) so test files can import simlar
at module level without the proprietary engine installed.

Covers every name imported from simlar_engine at module level across src/simlar/.
"""

from __future__ import annotations

import json
import os
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


def _resolve_directory(directory, base_dir=None) -> Path:
    if ".." in Path(directory).parts:
        raise ValueError(f"directory '{directory}' contains '..' components, which is not allowed")
    d = Path(directory).resolve()
    if base_dir is not None:
        base = Path(base_dir).resolve()
        if not d.is_relative_to(base):
            raise ValueError(
                f"directory '{directory}' resolves to '{d}', which escapes the allowed "
                f"base directory '{base}'"
            )
    return d


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

    def add(self, ids, vectors, parallel=False):
        self._ids = list(ids)
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._trained = True

    def update(self, ids, vectors):
        pass

    def delete(self, ids):
        pass

    def update_vector(self, doc_id: int, vector):
        pass

    def search(self, query, k=10, parallel=False, batch_size=None, candidates=None):
        n = min(k, len(self._ids))
        return [_SearchResult(rank=i, id=self._ids[i], score=1.0 / (i + 1)) for i in range(n)]

    def search_raw(self, vectors, k, candidates=None, parallel=False, n_candidates=None):
        n = min(k, len(self._ids))
        return np.arange(n, dtype=np.int64), np.ones(n, dtype=np.float32)

    def save(self, directory, base_dir=None):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "simlar"})

    @classmethod
    def load(cls, directory, base_dir=None):
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
    def __init__(self, method="lucene", k1=1.5, b=0.75, stopwords_lang=None, stemmer_lang=None):
        self._ids: list[str] = []
        self._trained = False

    def fit(self, texts, parallel=False, params=None, **kwargs):
        self._trained = True

    def add(self, ids, texts, parallel=False):
        self._ids = list(ids)
        self._trained = True

    def update(self, ids, texts):
        pass

    def delete(self, ids):
        pass

    def search(self, query, k=10, parallel=False, batch_size=None, candidates=None):
        positions = (
            list(range(len(self._ids)))
            if candidates is None
            else [p for p in candidates if 0 <= p < len(self._ids)]
        )
        n = min(k, len(positions))
        return [
            _SearchResult(rank=i, id=self._ids[positions[i]], score=1.0 / (i + 1)) for i in range(n)
        ]

    def search_raw(self, query, k, parallel=False, candidates=None):
        positions = (
            list(range(len(self._ids)))
            if candidates is None
            else [p for p in candidates if 0 <= p < len(self._ids)]
        )
        n = min(k, len(positions))
        return np.array(positions[:n], dtype=np.int64), np.ones(n, dtype=np.float32)

    def save(self, directory, base_dir=None):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "bm25"})

    @classmethod
    def load(cls, directory, base_dir=None):
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
        speed_preference="balanced",
    ):
        self._ids: list[str] = []
        self._trained = False
        self.speed_preference = speed_preference
        self.expected_recall = None
        self._text_index = text_index or _RelevanceCore()
        self._vector_index = vector_index or _SimlarCore()
        self._fusion = fusion or _ReciprocalRankFusion()
        self._text_k = text_k
        self._vector_k = vector_k
        self._top_k = top_k

    def fit(self, corpus, vectors, parallel=False, params=None):
        self._trained = True

    def add(self, ids, texts=None, vectors=None, parallel=False):
        # Forward to the injected sub-indexes (2-arg add(), matching the TextIndex/VectorIndex
        # ABC contract) so tests can verify a custom text_index/vector_index actually receives
        # what was added, not just that construction didn't crash.
        if texts is not None:
            self._text_index.add(ids, texts)
        if vectors is not None:
            self._vector_index.add(ids, vectors)
        self._ids.extend(ids)
        self._trained = True

    def update(self, ids, texts=None, vectors=None):
        pass

    def delete(self, ids):
        doomed = set(ids)
        self._ids = [i for i in self._ids if i not in doomed]

    def search(
        self,
        query_text=None,
        query_vector=None,
        k=None,
        parallel=False,
        batch_size=None,
        candidates=None,
    ):
        effective_k = k or self._top_k
        n = min(effective_k, len(self._ids))
        return [_SearchResult(rank=i, id=self._ids[i], score=1.0 / (i + 1)) for i in range(n)]

    def save(self, directory, base_dir=None):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "helix"})

    @classmethod
    def load(cls, directory, base_dir=None):
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


class _HashMatchCore:
    def __init__(self, stopwords_lang="english"):
        self._ids: list[str] = []
        self._trained = False

    def fit(self, corpus, parallel=False, params=None, **kwargs):
        self._trained = True

    def add(self, ids, texts, parallel=False):
        self._ids = list(ids)
        self._trained = True

    def update(self, ids, texts):
        pass

    def delete(self, ids):
        self._ids = [i for i in self._ids if i not in set(ids)]

    def search(self, query, k=10, parallel=False, batch_size=None, candidates=None):
        positions = (
            list(range(len(self._ids)))
            if candidates is None
            else [p for p in candidates if 0 <= p < len(self._ids)]
        )
        n = min(k, len(positions))
        return [
            _SearchResult(rank=i, id=self._ids[positions[i]], score=1.0 / (i + 1)) for i in range(n)
        ]

    def search_raw(self, queries, k, parallel=False, candidates=None):
        positions = (
            list(range(len(self._ids)))
            if candidates is None
            else [p for p in candidates if 0 <= p < len(self._ids)]
        )
        n = min(k, len(positions))
        return np.array(positions[:n], dtype=np.int64), np.ones(n, dtype=np.float32)

    def save(self, directory, base_dir=None):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "lookup"})

    @classmethod
    def load(cls, directory, base_dir=None):
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

    def search(
        self,
        query_text=None,
        query_vector=None,
        k=10,
        parallel=False,
        batch_size=None,
        candidates=None,
    ):
        n = min(k if k is not None else 10, self._count)
        return (
            np.arange(n, dtype=np.int64),
            np.array([1.0 / (i + 1) for i in range(n)], dtype=np.float32),
        )

    def save(self, directory, base_dir=None):
        Path(directory).mkdir(parents=True, exist_ok=True)
        _write_config(Path(directory) / "config.json", {"index_type": "streaming"})

    @classmethod
    def load(cls, directory, base_dir=None):
        return cls()

    @property
    def size(self):
        return self._count

    @property
    def n_shards(self):
        return 1 if self._trained else 0

    @property
    def is_trained(self):
        return self._trained

    @property
    def index_type(self):
        return "streaming_hybrid"

    @property
    def boundaries(self):
        return None

    @property
    def fit_values(self):
        return None


# ── Stub filtering (mirrors private simlar_engine FilteredIndex / SQLFilter) ──
# Just enough for the integrations to build and search *unfiltered* on top of
# FilteredIndex: ids / metadata / roles are recorded, searches pass straight
# through. Actually filtering needs the real engine -- see
# tests/integrations/test_filtering_integrations.py (SIMLAR_REAL_ENGINE=1).


class _FilterError(ValueError):
    pass


class _MetadataFilter:
    pass


class _SQLFilter:
    def __init__(self):
        self._ids: list[str] = []
        self._meta: dict[str, dict] = {}
        self._roles: dict[str, set] = {}

    def add(self, ids, metadata=None):
        dupes = [i for i in ids if i in self._meta]
        if dupes:
            raise ValueError(f"IDs already in filter: {dupes}")
        for i, m in zip(ids, metadata or [{} for _ in ids], strict=False):
            self._ids.append(i)
            self._meta[i] = dict(m)

    def update(self, ids, metadata):
        for i, m in zip(ids, metadata, strict=False):
            self._meta[i].update(m)

    def delete(self, ids):
        for i in ids:
            self._meta.pop(i, None)
            self._roles.pop(i, None)
        self._ids = [i for i in self._ids if i in self._meta]

    def compact(self):
        pass

    def grant(self, ids, roles):
        for i in ids:
            self._roles.setdefault(i, set()).update(roles)

    def revoke(self, ids, roles):
        for i in ids:
            self._roles.get(i, set()).difference_update(roles)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self._ids,
            "meta": self._meta,
            "roles": {k: sorted(v) for k, v in self._roles.items()},
        }
        (d / "filter.json").write_text(json.dumps(payload))

    @classmethod
    def load(cls, directory):
        payload = json.loads((Path(directory) / "filter.json").read_text())
        obj = cls()
        obj._ids = payload["ids"]
        obj._meta = payload["meta"]
        obj._roles = {k: set(v) for k, v in payload["roles"].items()}
        return obj

    @property
    def ids(self):
        return list(self._ids)

    @property
    def size(self):
        return len(self._ids)


class _FilteredIndex:
    _load_inner = None

    def __init__(self, inner, sql_filter=None):
        self._inner = inner
        self._filter = sql_filter if sql_filter is not None else _SQLFilter()

    def add(self, ids, *args, metadata=None, roles=None, **kwargs):
        self._filter.add(ids, metadata)
        self._inner.add(ids, *args, **kwargs)
        for i, rs in zip(ids, roles or [], strict=False):
            self._filter.grant([i], rs)

    def _unfiltered(self, filter, roles, candidates):
        if filter is not None or roles is not None or candidates is not None:
            raise NotImplementedError("filtering needs the real simlar_engine")

    def search(self, *args, filter=None, roles=None, candidates=None, **kwargs):
        self._unfiltered(filter, roles, candidates)
        return self._inner.search(*args, **kwargs)

    def search_raw(self, *args, filter=None, roles=None, candidates=None, **kwargs):
        self._unfiltered(filter, roles, candidates)
        return self._inner.search_raw(*args, **kwargs)

    def candidates(self, filter=None, roles=None):
        self._unfiltered(filter, roles, None)
        return None

    def grant(self, ids, roles):
        self._filter.grant(ids, roles)

    def revoke(self, ids, roles):
        self._filter.revoke(ids, roles)

    def update(self, ids, *args, metadata=None, **kwargs):
        if args or kwargs:
            self._inner.update(ids, *args, **kwargs)
        if metadata is not None:
            self._filter.update(ids, metadata)

    def delete(self, ids):
        self._inner.delete(ids)
        self._filter.delete(ids)
        self._filter.compact()

    def update_metadata(self, ids, metadata):
        self._filter.update(ids, metadata)

    def save(self, directory, base_dir=None):
        d = _resolve_directory(directory, base_dir)
        self._inner.save(str(d / "inner"))
        self._filter.save(str(d / "filter"))

    @classmethod
    def load(cls, directory, base_dir=None):
        from simlar.indexes.helix_index import HelixIndex

        d = _resolve_directory(directory, base_dir)
        return cls(HelixIndex.load(str(d / "inner")), _SQLFilter.load(str(d / "filter")))

    @property
    def inner(self):
        return self._inner

    @property
    def sql_filter(self):
        return self._filter

    @property
    def ids(self):
        return self._filter.ids

    @property
    def size(self):
        return self._filter.size


# ── Inject stubs into sys.modules ─────────────────────────────────────────────


def _inject_engine_stubs() -> None:
    """Populate sys.modules with stub simlar_engine sub-modules."""
    if "simlar_engine" in sys.modules:
        return  # already installed (real engine or previously stubbed)
    if os.environ.get("SIMLAR_REAL_ENGINE") == "1":
        # Opt-in: run against the installed proprietary engine instead. Only
        # the tests written for it (test_filtering_integrations.py) are
        # expected to pass this way; the rest assume the stubs below.
        import simlar_engine  # noqa: F401

        return

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
        FilterError=_FilterError,
        MetadataFilter=_MetadataFilter,
        SQLFilter=_SQLFilter,
        FilteredIndex=_FilteredIndex,
    )
    _mod("simlar_engine._types", SearchResult=_SearchResult, _Parameters=_Parameters)
    _mod(
        "simlar_engine._persistence",
        FORMAT_VERSION=_FORMAT_VERSION,
        write_config=_write_config,
        read_config=_read_config,
        resolve_directory=_resolve_directory,
    )
    _mod("simlar_engine._registry")
    _mod("simlar_engine.fusion")
    _mod("simlar_engine.fusion.rrf", ReciprocalRankFusion=_ReciprocalRankFusion)
    _mod("simlar_engine.indexes")
    _mod("simlar_engine.indexes._simlar_impl", _SimlarCore=_SimlarCore)
    _mod("simlar_engine.indexes._helix_impl", _HelixCore=_HelixCore)
    _mod("simlar_engine.indexes._relevance_impl", _RelevanceCore=_RelevanceCore)
    _mod("simlar_engine.indexes._hash_match_impl", _HashMatchCore=_HashMatchCore)
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
