# Changelog

All notable changes to **simlar** (the open-source wrapper) are documented here.
Dates are in YYYY-MM-DD format.

## [1.2.0] — 2026-10-07

### Added
- `HelixIndex(speed_preference=...)`: "fastest" … "most accurate" (default "balanced").
  An unset `text_k` / `vector_k` is now tuned by the engine from the corpus size and `top_k`.
  New read-only `speed_preference` and `expected_recall` properties.
- Embedding management: `Embedder` protocol (`embed_documents` / `embed_query`, so LangChain
  `Embeddings` fit as-is) and `CallableEmbedder(fn, query_fn=None, normalize=, batch_size=)`.
  `SimlarEngine`, `HelixIndex` and `FilteredIndex` take `embedder=`, so `add` / `update` /
  `fit` / `search` work from text alone (`texts=` / `query_text=` on `SimlarEngine`). Embedders
  are not persisted; reattach with `load(..., embedder=)`, `load_from_directory(..., embedder=)`
  or the `embedder` property. Indexes without an embedder behave exactly as before.
- `FilteredIndex` metadata filtering and role-based access in the LangChain `SimlarVectorStore`:
  `filter=` (SQL WHERE fragment, LangChain-style dict or `MetadataFilter`) and `roles=` on every
  search method and through `as_retriever(search_kwargs=...)`; `roles=` on `add_texts` /
  `add_documents` / `from_texts`; `grant()` / `revoke()`.
- LangChain `SimlarVectorStore.add_documents`, `delete`, `get_by_ids`,
  `similarity_search_by_vector`, `similarity_search_with_score_by_vector` and relevance scores
  (langchain-core 1.x interface).
- LlamaIndex `SimlarVectorStore` rebuilt on `FilteredIndex`: `MetadataFilters` (all operators
  incl. `contains` / `any` / `all` / `text_match` / `is_empty`, nested `and` / `or` / `not`),
  `doc_ids` / `node_ids`, query modes `default` (vector), `sparse` / `text_search` (BM25) and
  `hybrid`, `roles` via `vector_store_kwargs`, `delete` by `ref_doc_id`, `delete_nodes`,
  `get_nodes`, `clear`, full node round-trip (metadata, relationships), and
  `StorageContext.persist()` / `from_persist_dir()` compatibility.
- `HelixIndex.update()` / `HelixIndex.delete()` (requires simlar-engine with `_HelixCore.update/delete`).
- Haystack `SimlarDocumentStore` rebuilt on `FilteredIndex(HelixIndex)` (Haystack 3.x):
  - Haystack `filters` are resolved in SQL before ranking, so results are the exact top-k among
    matching documents. Semantics match Haystack's reference (`None` for missing fields, ISO-date
    comparisons, `FilterError` on invalid filters): exact conditions run as one cached SQL query,
    the rest are narrowed in SQL and checked with `document_matches_filter`.
  - `roles=` on `write_documents`, `filter_documents` and every retrieval method; `grant()` /
    `revoke()`.
  - `embedding_retrieval`, `bm25_retrieval` and `hybrid_retrieval`.
  - The full extended protocol: `delete_all_documents`, `delete_by_filter`, `update_by_filter`,
    `count_documents_by_filter`, `count_unique_metadata_by_filter`, `get_metadata_fields_info`,
    `get_metadata_field_min_max` and `get_metadata_field_unique_values`.
  - Passes Haystack's `haystack.testing.document_store` suite.
- Haystack `SimlarEmbeddingRetriever` and `SimlarBM25Retriever` components.
  `SimlarHybridRetriever` gains `filters`, `filter_policy` (`REPLACE` / `MERGE`), `roles` and
  `return_embedding`. All three serialize with `to_dict` / `from_dict` and work with
  `Pipeline.dumps()` / `loads()` (`simlar.integrations.haystack` is added to Haystack's
  deserialization allowlist on import).

### Added
- `HelixIndex(speed_preference=...)`: "fastest" … "most accurate" (default "balanced").
  An unset `text_k` / `vector_k` is now tuned by the engine from the corpus size and `top_k`.
  New read-only `speed_preference` and `expected_recall` properties.
- Embedding management: `Embedder` protocol (`embed_documents` / `embed_query`, so LangChain
  `Embeddings` fit as-is) and `CallableEmbedder(fn, query_fn=None, normalize=, batch_size=)`.
  `SimlarEngine`, `HelixIndex` and `FilteredIndex` take `embedder=`, so `add` / `update` /
  `fit` / `search` work from text alone (`texts=` / `query_text=` on `SimlarEngine`). Embedders
  are not persisted; reattach with `load(..., embedder=)`, `load_from_directory(..., embedder=)`
  or the `embedder` property. Indexes without an embedder behave exactly as before.
- `FilteredIndex` metadata filtering and role-based access in the LangChain `SimlarVectorStore`:
  `filter=` (SQL WHERE fragment, LangChain-style dict or `MetadataFilter`) and `roles=` on every
  search method and through `as_retriever(search_kwargs=...)`; `roles=` on `add_texts` /
  `add_documents` / `from_texts`; `grant()` / `revoke()`.
- LangChain `SimlarVectorStore.add_documents`, `delete`, `get_by_ids`,
  `similarity_search_by_vector`, `similarity_search_with_score_by_vector` and relevance scores
  (langchain-core 1.x interface).
- LlamaIndex `SimlarVectorStore` rebuilt on `FilteredIndex`: `MetadataFilters` (all operators
  incl. `contains` / `any` / `all` / `text_match` / `is_empty`, nested `and` / `or` / `not`),
  `doc_ids` / `node_ids`, query modes `default` (vector), `sparse` / `text_search` (BM25) and
  `hybrid`, `roles` via `vector_store_kwargs`, `delete` by `ref_doc_id`, `delete_nodes`,
  `get_nodes`, `clear`, full node round-trip (metadata, relationships), and
  `StorageContext.persist()` / `from_persist_dir()` compatibility.
- `HelixIndex.update()` / `HelixIndex.delete()` (requires simlar-engine with `_HelixCore.update/delete`).
- Haystack `SimlarDocumentStore` rebuilt on `FilteredIndex(HelixIndex)` (Haystack 3.x):
  - Haystack `filters` are resolved in SQL before ranking, so results are the exact top-k among
    matching documents. Semantics match Haystack's reference (`None` for missing fields, ISO-date
    comparisons, `FilterError` on invalid filters): exact conditions run as one cached SQL query,
    the rest are narrowed in SQL and checked with `document_matches_filter`.
  - `roles=` on `write_documents`, `filter_documents` and every retrieval method; `grant()` /
    `revoke()`.
  - `embedding_retrieval`, `bm25_retrieval` and `hybrid_retrieval`.
  - The full extended protocol: `delete_all_documents`, `delete_by_filter`, `update_by_filter`,
    `count_documents_by_filter`, `count_unique_metadata_by_filter`, `get_metadata_fields_info`,
    `get_metadata_field_min_max` and `get_metadata_field_unique_values`.
  - Passes Haystack's `haystack.testing.document_store` suite.
- Haystack `SimlarEmbeddingRetriever` and `SimlarBM25Retriever` components.
  `SimlarHybridRetriever` gains `filters`, `filter_policy` (`REPLACE` / `MERGE`), `roles` and
  `return_embedding`. All three serialize with `to_dict` / `from_dict` and work with
  `Pipeline.dumps()` / `loads()` (`simlar.integrations.haystack` is added to Haystack's
  deserialization allowlist on import).

### Fixed
- Metadata keys `position` / `id` no longer reach the SQL filter. `position` overwrote the row's
  position, silently dropping the document from filtered searches (all integrations). SQL keywords
  (`group`, `order`, ...) and integers beyond 64 bits no longer crash ingestion. These keys and
  values stay on the returned documents; LangChain / LlamaIndex can't filter on them, while
  Haystack filters them through `document_matches_filter`.
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
- `HelixIndex.save()` / `load()` keep a custom `ReciprocalRankFusion` (k, weights); a reloaded
  index used to fall back to the default fusion and rank differently. Saving with any other
  fusion strategy raises `TypeError`.

### Changed
- `HelixIndex` without a `fusion` now weights its default RRF by `alpha_text` / `alpha_vector`
  (default 0.1 : 1.0, previously ignored), matching `StreamingHybridIndex`'s defaults.
  Default hybrid rankings change; pass `fusion=ReciprocalRankFusion()` for the old equal weights.
- `StreamingHybridIndex` takes its search depths from the tuner only: `text_k` / `vector_k`
  are removed from its constructor (passing them raises `TypeError`). New `speed_preference`
  argument and `speed_preference` / `expected_recall` properties. Indexes saved with explicit
  depths still load; the saved values are ignored with a warning.
- Haystack `SimlarDocumentStore` writes go straight to the index. Overwrites update in place and
  deletes compact, with no tombstones and no second copy of every text. `save()` writes `index/`
  + `store.json`; directories saved by simlar 1.0 still load (their live documents are
  re-indexed).
- Haystack results carry `Document.score` and the original `Document.id`. `meta` no longer gets
  `score` / `rank` / `doc_id` added.
- Haystack filters follow Haystack's reference semantics: an unknown operator raises
  `FilterError` (it used to match nothing).
- Haystack `get_metadata_field_unique_values(metadata_field, search_term, from_, size, filters)`
  returns `(values, total)`, and `get_metadata_fields_info` reports `int` rather than `long`,
  per the Haystack 3 interface.
- `haystack` extra now requires `haystack-ai>=3.0,<4`.
- LangChain `SimlarVectorStore` writes are incremental: new ids are appended to the index and
  re-added ids are updated in place. The store no longer rebuilds the index on every write and no
  longer keeps a copy of every embedding (~250 MiB less Python memory at 20k x 384-dim documents).
- `save_local` writes `index/` + `docs.json` (no pickle). Metadata must be JSON-serializable.
- `langchain` extra now requires `langchain-core>=1.0`; `llama_index` extra requires
  `llama-index-core>=0.14`.
- LlamaIndex `SimlarVectorStore` follows LlamaIndex's query-mode convention (as `PGVectorStore`
  does): `default` is vector-only; pass `vector_store_query_mode="hybrid"` for BM25 + vector
  (previously `default` ran hybrid whenever `query_str` was set).

### Removed
- `simlar.integrations.langchain.langchain_retriever.SimlarRetriever` — use
  `store.as_retriever(search_kwargs={"k": ..., "filter": ..., "roles": ...})`.
- `simlar.integrations.llama_index.simlar_retriever.SimlarRetriever` — use
  `VectorStoreIndex.from_vector_store(store).as_retriever(...)`.
- LlamaIndex `SimlarVectorStore(index, id_to_text, parallel)` and `SimlarVectorStore.from_texts()` —
  construct with `SimlarVectorStore(text_k=..., vector_k=..., top_k=..., parallel=...)` and add
  nodes through `VectorStoreIndex` or `store.add(nodes)`. Stores persisted by the old class
  (`index/` + `id_to_text.json`, no metadata) cannot be loaded.
- Loading a store saved by simlar 1.0 (`sidecar.pkl`) now requires
  `load_local(..., allow_dangerous_deserialization=True)`, since it is a pickle.

### Changed
- Haystack `SimlarDocumentStore` writes go straight to the index. Overwrites update in place and
  deletes compact, with no tombstones and no second copy of every text. `save()` writes `index/`
  + `store.json`; directories saved by simlar 1.0 still load (their live documents are
  re-indexed).
- Haystack results carry `Document.score` and the original `Document.id`. `meta` no longer gets
  `score` / `rank` / `doc_id` added.
- Haystack filters follow Haystack's reference semantics: an unknown operator raises
  `FilterError` (it used to match nothing).
- Haystack `get_metadata_field_unique_values(metadata_field, search_term, from_, size, filters)`
  returns `(values, total)`, and `get_metadata_fields_info` reports `int` rather than `long`,
  per the Haystack 3 interface.
- `haystack` extra now requires `haystack-ai>=3.0,<4`.
- LangChain `SimlarVectorStore` writes are incremental: new ids are appended to the index and
  re-added ids are updated in place. The store no longer rebuilds the index on every write and no
  longer keeps a copy of every embedding (~250 MiB less Python memory at 20k x 384-dim documents).
- `save_local` writes `index/` + `docs.json` (no pickle). Metadata must be JSON-serializable.
- `langchain` extra now requires `langchain-core>=1.0`; `llama_index` extra requires
  `llama-index-core>=0.14`.
- LlamaIndex `SimlarVectorStore` follows LlamaIndex's query-mode convention (as `PGVectorStore`
  does): `default` is vector-only; pass `vector_store_query_mode="hybrid"` for BM25 + vector
  (previously `default` ran hybrid whenever `query_str` was set).

### Removed
- `simlar.integrations.langchain.langchain_retriever.SimlarRetriever` — use
  `store.as_retriever(search_kwargs={"k": ..., "filter": ..., "roles": ...})`.
- `simlar.integrations.llama_index.simlar_retriever.SimlarRetriever` — use
  `VectorStoreIndex.from_vector_store(store).as_retriever(...)`.
- LlamaIndex `SimlarVectorStore(index, id_to_text, parallel)` and `SimlarVectorStore.from_texts()` —
  construct with `SimlarVectorStore(text_k=..., vector_k=..., top_k=..., parallel=...)` and add
  nodes through `VectorStoreIndex` or `store.add(nodes)`. Stores persisted by the old class
  (`index/` + `id_to_text.json`, no metadata) cannot be loaded.
- Loading a store saved by simlar 1.0 (`sidecar.pkl`) now requires
  `load_local(..., allow_dangerous_deserialization=True)`, since it is a pickle.

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
