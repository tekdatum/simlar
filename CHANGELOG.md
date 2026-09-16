# Changelog

All notable changes to **simlar** (the open-source wrapper) are documented here.
Dates are in YYYY-MM-DD format.

## [Unreleased]

### Fixed
- `LookupIndex.search_raw()` defaulted `parallel` to `True`, inconsistent with
  the abstract `TextIndex.search_raw` contract's `False` default and with its
  sibling `RelevanceIndex.search_raw()`. `search_raw` is an internal method
  whose only real caller (`HelixIndex`'s fusion path) always passes `parallel`
  explicitly, so the divergence had no principled reason behind it — aligned
  to `False`.
- `HelixIndex.__repr__()` read `self._core._text_k`/`_vector_k`, private
  Cython attributes that were never actually reachable from Python — calling
  `repr()` on any `HelixIndex` raised `AttributeError`. Requires
  `simlar-engine >= 1.1.0`'s new `_HelixCore.text_k`/`.vector_k` properties.
- `CompositeIndex.search()`'s declared return type (`list[SearchResult]`) was
  narrower than what `HelixIndex.search()` actually returns for a batch query
  (`list[SearchResult] | list[list[SearchResult]]`, the same batch contract
  `TextIndex.search()` already declares) — widened to match, and to accept
  `query_text` as `str | list[str]`.

## [1.1.0] — 2026-09-02

### Added
- `batch_size` parameter on `SimlarEngine.search()`, `HelixIndex.search()`,
  `RelevanceIndex.search()`, `LookupIndex.search()`, and
  `StreamingHybridIndex.search()`, passed through to the underlying engine
  core. Bounds the peak memory of a large query batch by chunking the query
  axis; `None` (the default) derives a chunk width from a memory budget, and
  results are unaffected by it either way — it's a memory strategy, not a
  change in ranking. Requires `simlar-engine >= 1.1.0`.
- `StreamingHybridIndex` now exposes `.size`, `.is_trained`, `.index_type`,
  `.n_shards`, `.boundaries`, and `.fit_values` — the same metadata contract
  every other index type already had.
- A best-effort `simlar_engine` version check at import time, so an
  incompatible engine build fails with a clear error instead of a confusing
  `TypeError` deep inside a `search()` call. See `docs/installation.md` for
  the compatible version matrix.
- `docs/installation.md`: installation instructions and the
  `simlar`/`simlar-engine` compatibility matrix.
- `LookupIndex` documentation (README, `docs/api-reference.md`,
  `docs/concepts.md`) — it shipped in 1.0.0 but was never documented.

### Fixed
- `LookupIndex.search()`'s `batch_size` argument was accepted but never
  forwarded to the engine core, so it silently had no effect. It is now
  passed through like the other index classes.
- `StreamingHybridIndex.search()` didn't accept `batch_size` at all, even
  though the underlying engine core already supported chunking it.

### Removed
- `LookupIndex.__init__`'s `stemmer_lang` parameter. `LookupIndex` is a pure
  term-match index with no stemming concept, so the parameter was always a
  silent no-op.

### Changed
- `parallel` now defaults to `True` on `add()`/`fit()`/`search()` across
  `SimlarEngine`, `HelixIndex`, `RelevanceIndex`, and `LookupIndex`. The
  public `.pyi` type stubs (`_simlar.pyi`, `_helix.pyi`, `_relevance.pyi`,
  `_streaming.pyi`) had drifted from this and other signature changes —
  stale `parallel` defaults, a nonexistent `rrf_k` parameter on `HelixIndex`,
  a nonexistent `update_vector` method on `SimlarEngine`, a nonexistent
  `fit()` on `StreamingHybridIndex` — and have been resynced against the
  real implementations.

## [1.0.0] — 2026-06-24

### Added
- Initial public release.
- `SimlarEngine` — semantic vector index backed by the proprietary binary engine.
- `RelevanceIndex` — Text search index.
- `HelixIndex` — hybrid index fusing keyword and vector signals via RRF.
- `StreamingHybridIndex` — streaming hybrid index for large corpora added in batches.
- `ReciprocalRankFusion` fusion strategy.
- LangChain `SimlarVectorStore` integration (`pip install simlar[langchain]`).
- Haystack integration (`pip install simlar[haystack]`).
- `save` / `load_from_directory` persistence interface.
- `@register` extension point for custom index types.
- `SearchResult`, `IndexConfig`, `TextIndex`, `VectorIndex` public contracts.
