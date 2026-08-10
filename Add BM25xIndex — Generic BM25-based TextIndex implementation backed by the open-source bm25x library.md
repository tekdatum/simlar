**Component:** `simlar` (public wrapper only)

---

Introduce a new `TextIndex` implementation, `BM25xIndex`, backed by the open-source `bm25x` Rust library.

Unlike `RelevanceIndex`, this implementation must not depend on `simlar_engine` and should provide a completely independent keyword-search backend while fully implementing the existing `TextIndex` contract.

The goal is to make **any text index** pluggable behind the existing abstraction, using `bm25x` as the first external implementation.

---

# Background

Today the only keyword-search implementation available in `simlar` is `RelevanceIndex`, which delegates all scoring to the proprietary `simlar_engine`.

`bm25x` already provides:

- native Rust implementation
- incremental indexing
- update/delete support
- persistence
- mmap loading
- optional GPU support
- built-in stemming/tokenization

Rather than implementing BM25 internally, `simlar` should wrap this library exactly as it wraps other indexing engines.

---

# Problem Statement

The current `TextIndex` abstraction assumes:

- caller-supplied string IDs
- `SearchResult` return values
- registry-based persistence
- common serialization format

`bm25x` instead exposes:

- integer document IDs
- tuple-based search results
- native directory persistence
- no knowledge of `simlar` registry metadata

An adapter layer is therefore required.

---

# Goals

- Implement `BM25xIndex(TextIndex)`
- Register under `"bm25x"`
- No dependency on `simlar_engine`
- Preserve the current `TextIndex` API
- Support:
  - fit
  - add
  - update
  - delete
  - search
  - search_raw
  - save/load
- Maintain external string ID mapping
- Reuse existing registry/persistence mechanisms
- Work transparently anywhere a `TextIndex` is accepted

#

---

# Proposed API

```
@register("bm25x")
class BM25xIndex(TextIndex):

    def __init__(
        self,
        method="lucene",
        k1=1.5,
        b=0.75,
        delta=0.5,
    ):
        ...

    def fit(...)
    def add(...)
    def update(...)
    def delete(...)
    def search(...)
    def search_raw(...)
    def save(...)
    @classmethod
    def load(...)
    @property
    def size(self) -> int: ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def index_type(self) -> str: ...    # "bm25x"
    @property
    def ids(self) -> list[str]: ...

```

---

# Required Work

## New implementation

- `src/simlar/indexes/bm25x_index.py`

## Public exports

- `src/simlar/indexes/__init__.py`
- `src/simlar/__init__.py`

## Dependency

Add:

- `bm25x`

Update:

- `pyproject.toml`
- mypy overrides

## Tests

Create:

- `tests/test_bm25x_index.py`

Update CI to install `bm25x`.

---

# Adapter Responsibilities

## ID bookkeeping

Maintain:

- external string IDs
- internal integer document positions

Provide:

```
external ID
      ↓
integer position
      ↓
bm25x
```

---

## Search conversion

Convert

```
(doc_id, score)
```

into

```
SearchResult(
    rank=...,
    id=...,
    score=...
)
```

---

## search_raw()

Convert native tuples into

```
(np.ndarray[int64], np.ndarray[float64])
```

matching the existing contract.

---

## Persistence

Wrapper persistence should include:

```
config.json
ids.json
bm25x_native/
```

using the existing registry envelope plus the library's own serialization.

---

# Integration

No changes required to:

- Registry
- HelixIndex
- StreamingHelixIndex
- persistence APIs

`BM25xIndex` should function anywhere a `TextIndex` is accepted.

---

# Error Handling

Validate before calling `bm25x`:

- duplicate IDs
- unknown IDs
- empty corpus
- invalid updates
- invalid deletes

Propagate library errors for unsupported constructor options.

---

# Testing

Cover:

- construction
- fit
- add
- update
- delete
- search
- search_raw
- persistence round-trip
- registry loading
- Helix integration
- StreamingHelix integration
- edge cases
- CI execution using the real `bm25x` package

---

# Documentation

Update:

- README
- API Reference
- Concepts
- CHANGELOG

Document:

- incremental indexing
- dependency installation
- Apache-2.0 licensing
- differences from `RelevanceIndex`

---

# Acceptance Criteria

- `from simlar import BM25xIndex` works
- Registered as `"bm25x"`
- Fully implements `TextIndex`
- Incremental add works
- ID mapping behaves correctly
- Search returns `SearchResult`
- `search_raw()` returns NumPy arrays
- Save/load round-trips successfully
- Compatible with `HelixIndex`
- Compatible with `StreamingHelixIndex`
- CI passes across supported Python versions
- Documentation updated
- Dependency and licensing documented

---
