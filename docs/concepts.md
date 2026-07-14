# Concepts

## What is simlar?

`simlar` is a search library that finds documents similar to a query. It combines two complementary signals — keyword relevance and semantic meaning — and merges them into a single ranked result.

## Adding documents

There are two ways to add documents to an index:

| Method | When to use |
|--------|-------------|
| `fit(corpus)` | You only care about the order of results, not named IDs |
| `add(ids, corpus)` | You want results mapped back to your own document identifiers |

Both methods accept raw text or pre-computed vectors.

```python
from simlar import RelevanceIndex

# fit(corpus): documents get auto-assigned string ids "0", "1", "2", ...
idx = RelevanceIndex()
idx.fit(["hello world", "foo bar baz"])
results = idx.search("hello", k=1)
results[0].id  # "0"

# add(ids, corpus): documents are mapped to the ids you provide
idx = RelevanceIndex()
idx.add(ids=["a", "b"], texts=["hello world", "foo bar baz"])
results = idx.search("hello", k=1)
results[0].id  # "a"
```

`fit()` and `add()` apply to `RelevanceIndex`, `SimlarEngine`, and `HelixIndex`. `StreamingHybridIndex` does not have either method — see [below](#streaminghybridindex--hybrid-search-at-scale) for how it ingests documents.

## Indexes

simlar ships four index types. Pick the one that matches your data and scale.

### RelevanceIndex — keyword search

Finds documents that contain the right words. It understands that a rare word is a stronger signal than a common one, and it accounts for document length so short and long documents are compared fairly.

**Use when:** your queries are keyword-based and you need fast, explainable results. No embeddings required.

**Limitation:** does not understand meaning — searching for "car" won't surface a document that only says "automobile."

---

### SimlarEngine — semantic search

Finds documents that mean the same thing as your query, even if they use different words. It works on vector embeddings you provide, comparing the direction and proximity of meaning in a high-dimensional space.

**Use when:** you have pre-computed embeddings and need semantic similarity — for example, matching questions to answers, or finding paraphrases.


---

### HelixIndex — hybrid search (text + vectors, in memory)

Combines keyword relevance and semantic similarity into a single ranked result. It runs both signals in parallel and merges the ranked lists, so a document scores well if it matches on either dimension — or both.

**Use when:** you want the best of both worlds: exact-word precision and semantic depth, on a corpus that fits comfortably in memory.

---

### StreamingHybridIndex — hybrid search at scale

The same hybrid approach as HelixIndex, but designed for corpora too large to fit in memory at once. Documents are added in batches via `add_batch(corpus, vectors, parallel=False)`; each batch is stored as an independent shard. At query time all shards are searched and the results are merged globally.

**Use when:** you are ingesting millions of documents in a streaming fashion, or when your corpus would exhaust available RAM with a standard index.

**Note:** unlike the other indexes, `StreamingHybridIndex` does not have `fit()` or `add()` — use `add_batch()` instead.

---

### Which one to pick

| I have… | I need… | Index |
|---------|---------|-------|
| Text | Keyword matching | `RelevanceIndex` |
| Embeddings | Semantic similarity | `SimlarEngine` |
| Text + embeddings | Both signals | `HelixIndex` |
| Text + embeddings, massive scale | Both signals at scale | `StreamingHybridIndex` |

---

## How search works (conceptually)

When you run a query, the index scores every document twice — once for keyword overlap, once for semantic closeness — and then blends the two ranked lists into one. You can control how much weight each signal carries.

## Scaling to large collections

For very large corpora, documents are split into groups (shards) that are searched independently and then merged. This keeps memory usage bounded while still returning globally ranked results.

## Saving and loading

Every index can be saved to a directory and restored later:

```python
index.save("/path/to/dir")

index = load_from_directory("/path/to/dir")
```

The loader figures out the index type automatically — you don't need to remember which class you used.

## Extending simlar

You can plug in your own index implementation and it will behave like any built-in index — including save/load support:

```python
from simlar import VectorIndex, register

@register("my_index")
class MyIndex(VectorIndex):
    ...
```
