# Changelog

All notable changes to **simlar** (the open-source wrapper) are documented here.
Dates are in YYYY-MM-DD format.

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
