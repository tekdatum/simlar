# Changelog

All notable changes to **simlar** (the open-source wrapper) are documented here.
Dates are in YYYY-MM-DD format.

## [Unreleased]

### Added
- `BM25xIndex` — a `TextIndex` implementation backed by the open-source
  [`bm25x`](https://pypi.org/project/bm25x/) library, with no dependency on `simlar_engine`.
  Registered as `"bm25x"`; drop-in compatible with `HelixIndex`/`StreamingHybridIndex` anywhere
  a `TextIndex` is accepted. Install via `pip install "simlar[bm25x]"` — currently gated to
  **Python 3.12** by an environment marker, since `bm25x` only ships wheels for that version.

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
