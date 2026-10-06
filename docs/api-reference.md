# API Reference

All public names are importable from the top-level `simlar` package:

```python
from simlar import (
    SearchResult,
    RelevanceIndex, SimlarEngine, HelixIndex, StreamingHybridIndex, LookupIndex,
    ReciprocalRankFusion,
    register, load_from_directory,
)
```

---

## SearchResult

A single result returned by `search()`.

```python
@dataclass
class SearchResult:
    rank:  int
    id:    str
    score: float
    text:  str | None = None
```

`text` is populated only when the index stored the original document text.

---

## RelevanceIndex

Keyword search over text documents, ranked by BM25 relevance (term rarity and document length are both accounted for).

```python
class RelevanceIndex:
    def __init__(
        self,
        method: str = "robertson",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords_lang: str = "english",
        stemmer_lang: str = "english",
    ) -> None: ...

    def add(self, ids: list[str], texts: list[str], parallel: bool = True) -> None: ...
    def search(self, query: str | list[str], k: int = 10, parallel: bool = True, batch_size: int | None = None) -> list[SearchResult]: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> RelevanceIndex: ...
```

`search()` accepts a single query string or a list of them; a batch runs as one call rather than one per query. `batch_size` bounds the peak memory of a very large batch by chunking the query axis — it never changes the ranking, only how much memory one call uses at a time.

**Note:** Adding new documents replaces the entire index. To append, pass the full combined corpus to `add()`.

### Example

```python
from simlar import RelevanceIndex

idx = RelevanceIndex()
idx.add(ids=["a", "b", "c"], texts=["hello world", "foo bar baz", "quick brown fox"])

results = idx.search("hello", k=2)
# results[0].id == "a"
```

---

## SimlarEngine

Semantic search over vector embeddings.

```python
class SimlarEngine:
    def __init__(self, n_candidates: int | None = None) -> None: ...

    def add(self, ids: list[str], vectors: np.ndarray, parallel: bool = True) -> None: ...
    def update(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search(self, query: np.ndarray, k: int = 10, parallel: bool = True, batch_size: int | None = None) -> list[SearchResult]: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> SimlarEngine: ...
```

`n_candidates` sizes the internal shortlist; leave it `None` to size it automatically from your corpus. Supports incremental `add()`, `update()`, and `delete()` after the initial fit. `batch_size` bounds the peak memory of a large query batch by chunking the query axis, same as `RelevanceIndex.search()`.

### Example

```python
import numpy as np
from simlar import SimlarEngine

idx = SimlarEngine()
vecs = np.random.rand(1000, 128).astype(np.float32)
idx.add(ids=[str(i) for i in range(1000)], vectors=vecs)

query = np.random.rand(1, 128).astype(np.float32)
results = idx.search(query, k=10)
```

---

## HelixIndex

Hybrid search — combines keyword and semantic results into one ranked list.

```python
class HelixIndex:
    def __init__(
        self,
        *,
        text_index: TextIndex | None = None,
        vector_index: VectorIndex | None = None,
        fusion: ReciprocalRankFusion | None = None,
        text_k: int | None = None,
        vector_k: int | None = None,
        top_k: int = 100,
        alpha_text: float = 0.10,
        alpha_vector: float = 1.0,
    ) -> None: ...

    def add(
        self,
        ids: list[str],
        texts: list[str] | None = None,
        vectors: np.ndarray | None = None,
        parallel: bool = True,
    ) -> None: ...

    def search(
        self,
        query_text: str | list[str] | None = None,
        query_vector: np.ndarray | None = None,
        k: int | None = None,
        parallel: bool = True,
        batch_size: int | None = None,
    ) -> list[SearchResult]: ...

    def save(self, directory: str) -> None: ...

    @classmethod
    def load(cls, directory: str) -> HelixIndex: ...
```

- Pass only `query_text` to search by keywords, only `query_vector` for semantics, or both for full hybrid search.
- `alpha_text` and `alpha_vector` control how much each signal influences the final ranking.

### Example

```python
from simlar import HelixIndex, RelevanceIndex, SimlarEngine, ReciprocalRankFusion

index = HelixIndex(
    text_index=RelevanceIndex(),
    vector_index=SimlarEngine(),
    fusion=ReciprocalRankFusion(),
    top_k=20,
)
index.add(ids=ids, texts=corpus, vectors=vectors)

results = index.search(query_text="query", query_vector=q_vec, k=10)
```

---

## LookupIndex

Exact keyword matching — a lighter-weight alternative to `RelevanceIndex` when you want term overlap only, with no relevance ranking (no IDF, no length normalization, no stemming).

```python
class LookupIndex:
    def __init__(self, stopwords_lang: str = "english") -> None: ...

    def add(self, ids: list[str], texts: list[str], parallel: bool = True) -> None: ...
    def search(self, query: str | list[str], k: int = 10, parallel: bool = True, batch_size: int | None = None) -> list[SearchResult]: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> LookupIndex: ...
```

A document's score is the number of *distinct* query terms it contains — there's no notion of one match mattering more than another. Pass `stopwords_lang=None` to disable stopword filtering and keep every term.

**Use when:** you want fast exact-term filtering (e.g. as a pre-filter or a cheap fallback) and don't need BM25-style relevance ranking — for ranked keyword search, use `RelevanceIndex` instead.

### Example

```python
from simlar import LookupIndex

idx = LookupIndex()
idx.add(ids=["a", "b"], texts=["hello world", "foo bar"])

results = idx.search("hello", k=5)
# results[0].id == "a"
```

---

## StreamingHybridIndex

Hybrid search for large corpora that are added in batches. Documents are split into shards as they arrive; a search retrieves from every shard and merges the results globally.

```python
class StreamingHybridIndex:
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
    ) -> None: ...

    def add_batch(
        self,
        corpus: list[str],
        vectors: np.ndarray,
        parallel: bool = False,
    ) -> None: ...

    async def add_batch_async(
        self,
        corpus: list[str],
        vectors: np.ndarray,
        parallel: bool = False,
    ) -> None: ...

    def search(
        self,
        query_text: str | list[str],
        query_vector: np.ndarray,
        k: int | None = None,
        parallel: bool = False,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def save(self, directory: str) -> None: ...

    @classmethod
    def load(cls, directory: str) -> StreamingHybridIndex: ...
```

`search()` returns `(ids, scores)` — both `np.ndarray` of shape `(k,)`. `ids` are integer positions into the original corpus; retrieve the document with `corpus[int(ids[i])]`. Unlike the other index types, `StreamingHybridIndex` has no `fit()`/`add()` — ingest exclusively through `add_batch()`. It exposes the same metadata properties as the other indexes (`.size`, `.is_trained`, `.index_type`, `.boundaries`, `.fit_values`), plus `.n_shards`.

### Example

```python
import numpy as np
from simlar import StreamingHybridIndex

index = StreamingHybridIndex(top_k=20)

for batch_texts, batch_vecs in stream_corpus():
    index.add_batch(batch_texts, batch_vecs)

ids, scores = index.search("query text", query_vector=q_vec, k=20)
```

---

## ReciprocalRankFusion

Merges multiple ranked result lists into one. Used as the `fusion=` argument in `HelixIndex`.

```python
class ReciprocalRankFusion:
    def __init__(
        self,
        k: int = 2,
        weights: list[float] | None = None,
    ) -> None: ...
```

`k` is the RRF rank-damping constant. `weights` control how much each signal contributes, and align with the order of indexes passed to `HelixIndex` — text first, vector second.

```python
# Give the vector signal 3× the weight of text
fusion = ReciprocalRankFusion(weights=[1.0, 3.0])
```

---

## Extending simlar

### register

Registers a custom index class so it works with `load_from_directory()` and the rest of the ecosystem.

```python
from simlar import register, VectorIndex

@register("my_index")
class MyIndex(VectorIndex):
    ...
```

### load_from_directory

Loads any saved index without knowing which type it is.

```python
from simlar import load_from_directory

index = load_from_directory("/path/to/saved/index")
```
