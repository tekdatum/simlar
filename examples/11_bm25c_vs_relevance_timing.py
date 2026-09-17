#!/usr/bin/env python
"""bm25c vs bm25x vs RelevanceIndex (bm25s) -- build/search timing comparison.

Deliberately has none of examples/10_rags_benchmark.py's RAG/framework layer
(no LangChain/LlamaIndex/Haystack, no embeddings, no LLM) -- all three are
plain TextIndex implementations, so this compares them directly: index build
time (fit), batched search time (search_raw, not a per-query loop -- every
engine's batch kernel is built for exactly this: bm25c_search_batch runs each
query on its own OpenMP thread, bm25x's search_batch is rayon-parallel
internally, and _RelevanceCore's search_raw does the equivalent via bm25s'
numba backend), and retrieval-quality (Hit@1/Hit@k/MRR) using the same
SquadDataset ground truth 10_rags_benchmark.py's test_retrievers uses for its
own Build(ms)/Query(ms) columns -- just without the retriever/RAG wrapping in
between.

Run across several k values, not just one: the point of this script is to
find where bm25c/bm25x lose ground to bm25s, and each engine's own
architecture makes it fast at a different k range --
  - bm25c: TAAT dense-accumulate, adaptive top-k (sorted-insertion below
    ~512, bucket-histogram above) -- roughly flat cost across k.
  - bm25x: DAAT + WAND pruning -- cheap when k is small/selective, but
    degrades as k grows because WAND's score-bound skipping stops paying
    off (there IS a quickselect-based `search_unordered` fix for this in
    the Rust crate already, but it is NOT yet exposed through the `bm25x`
    Python package or BM25xIndex -- search_raw always goes through the
    WAND path regardless of k, so this benchmark's large-k numbers for
    bm25x show exactly the gap that fix would close).
  - bm25s (RelevanceIndex): also TAAT dense-accumulate, like bm25c, but
    top-k is a binary heap (O(n log k)) instead of a bucket histogram, and
    the entire per-posting score (not just the length norm) is precomputed
    at fit time.
A single k=10 run mostly hides these differences; sweeping k is what
surfaces them.
"""

import time
from dataclasses import dataclass

from datasets import load_dataset

from simlar import BM25CIndex, BM25xIndex, RelevanceIndex


@dataclass
class SquadDataset:
    corpus: list[str]
    queries: list[str]
    query_to_context: dict[str, str]

    @classmethod
    def from_hf(cls, n_questions: int | None = None, n_contexts: int | None = None) -> "SquadDataset":
        hf = load_dataset("rajpurkar/squad", split="train")
        seen, corpus, queries, query_to_context = set(), [], [], {}
        for row in hf:
            if n_questions is not None and len(queries) == n_questions:
                break
            ctx = row["context"]
            if ctx not in seen:
                if n_contexts is not None and len(corpus) >= n_contexts:
                    continue
                seen.add(ctx)
                corpus.append(ctx)
            queries.append(row["question"])
            query_to_context[row["question"]] = ctx
        return cls(corpus=corpus, queries=queries, query_to_context=query_to_context)


def benchmark(name: str, index, dataset: SquadDataset, k: int, build_ms: float) -> dict:
    """Times one batched search_raw() call across every query at a given k
    -- not a per-query loop, so this measures each engine's actual batch
    throughput rather than Python call overhead repeated n times. `index`
    must already be fit()'d -- build time doesn't depend on k, so the
    caller times it once and passes it through rather than repeating it."""
    # One untimed warm-up query first: RelevanceIndex's bm25s backend JIT-
    # compiles via numba on its first call, which would otherwise get
    # attributed to "average query time" rather than being the one-time
    # cost it actually is. bm25c and bm25x have no such warm-up (already-
    # compiled C/Rust), so this is a no-op cost for them either way.
    index.search_raw(dataset.queries[:1], k=k, parallel=True)

    n = len(dataset.queries)
    t0 = time.perf_counter()
    ids, _scores = index.search_raw(dataset.queries, k=k, parallel=True)
    query_ms = (time.perf_counter() - t0) * 1000 / n

    # fit() auto-assigns ids as str(position), so a raw position doubles as
    # an index into dataset.corpus directly. -1 is the explicit padding
    # sentinel both bm25c and bm25x use for "fewer than k real matches";
    # RelevanceIndex has no such sentinel (bm25s's own top-k fills unmatched
    # slots with an arbitrary real position instead -- see
    # _relevance_impl.pyx) -- harmless here, since a bogus padding position
    # just won't match the correct context below, so it never produces a
    # false hit.
    hit1 = hitk = mrr = 0.0
    for q, row in zip(dataset.queries, ids):
        correct = dataset.query_to_context.get(q)
        rank = next((i for i, pos in enumerate(row) if pos >= 0 and dataset.corpus[pos] == correct), None)
        if rank is not None:
            if rank == 0:
                hit1 += 1
            hitk += 1
            mrr += 1 / (rank + 1)

    return {
        "name": name,
        "build_ms": build_ms,
        "query_ms": query_ms,
        "hit1": hit1 / n,
        "hitk": hitk / n,
        "mrr": mrr / n,
    }


def main() -> None:
    print("Loading dataset...")
    # n_contexts controls corpus size directly (SQuAD averages ~4-5 questions
    # per context, so n_questions alone is an unpredictable proxy for corpus
    # size -- 200 questions collapsed to just 41 unique contexts earlier).
    dataset = SquadDataset.from_hf(n_contexts=20000)
    n_docs = len(dataset.corpus)
    print(f"Dataset: {n_docs} documents & {len(dataset.queries)} queries\n")

    # Build once per engine, reuse across every k in the sweep below --
    # fit() cost doesn't depend on k, so timing it k_values-many times would
    # just repeat the same number.
    engines = {
        "bm25c": BM25CIndex(),
        "bm25x": BM25xIndex(),
        "relevance": RelevanceIndex(),
    }
    build_ms = {}
    for name, index in engines.items():
        t0 = time.perf_counter()
        index.fit(dataset.corpus)
        build_ms[name] = (time.perf_counter() - t0) * 1000

    header = f"{'Index':<12}{'k':>8}{'Build(ms)':>12}{'Query(ms)':>12}{'Hit@1':>10}{'Hit@10':>10}{'MRR':>8}"
    for k in (10, 100, 1000, 5000, n_docs):
        print(header)
        print("-" * len(header))
        for name, index in engines.items():
            r = benchmark(name, index, dataset, k=k, build_ms=build_ms[name])
            print(
                f"{r['name']:<12}{k:>8}{r['build_ms']:>12.0f}{r['query_ms']:>12.3f}"
                f"{r['hit1']:>10.1%}{r['hitk']:>10.1%}{r['mrr']:>8.3f}"
            )
        print()


if __name__ == "__main__":
    main()
