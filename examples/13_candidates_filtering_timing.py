#!/usr/bin/env python
"""SQLFilter candidates -- build/search timing comparison across selectivity.

Same shape as examples/11_bm25c_vs_relevance_timing.py (plain TextIndex
build/search timing, no RAG/framework layer), but the axis under test here is
filtering, not indexing engine: for all four TextIndex backends -- this
measures query time and retrieval quality at a few different filter
selectivities against the exact same corpus and query set.

Every backend supports `candidates=` as a kwarg directly on search_raw() now
(formalized on TextIndex itself, in contracts.py) -- a single flat array of
positions applied to every query in a batch call.

One SQLFilter (simlar_engine._sql_filter.SQLFilter -- a standalone SQL-driven
metadata pre-filter, not itself a TextIndex), reused for all four backends:
every backend assigns positions by strict append order, and all get fit() on
the identical corpus list in the identical order, so their positions align
by construction (same alignment argument as HelixIndex's own text/vector
sub-indexes). SQLFilter is built the same way -- add()'d with the same
auto-generated str(i) ids in the same order -- so it stays aligned with
every backend without needing to be paired with any of them.

Quality (Hit@1/Hit@10/MRR) is computed only over queries whose correct
context survives the filter ("in scope") -- the synthetic category assigned
to each doc here is arbitrary (i % N_CATEGORIES), not query-aware, so scoring
it against ALL queries would just measure how often the random filter
happens to keep the right answer, not whether filtered search still finds
it correctly when it's actually a candidate.
"""

import time
from dataclasses import dataclass

from datasets import load_dataset

from simlar import BM25CIndex, BM25xIndex, LookupIndex, RelevanceIndex
from simlar_engine import SQLFilter

N_CATEGORIES = 10


@dataclass
class SquadDataset:
    corpus: list[str]
    queries: list[str]
    query_to_context: dict[str, str]

    @classmethod
    def from_hf(cls, n_contexts: int | None = None) -> "SquadDataset":
        hf = load_dataset("rajpurkar/squad", split="train")
        seen, corpus, queries, query_to_context = set(), [], [], {}
        for row in hf:
            ctx = row["context"]
            if ctx not in seen:
                if n_contexts is not None and len(corpus) >= n_contexts:
                    continue
                seen.add(ctx)
                corpus.append(ctx)
            queries.append(row["question"])
            query_to_context[row["question"]] = ctx
        return cls(corpus=corpus, queries=queries, query_to_context=query_to_context)


def benchmark_scenario(name: str, index, dataset: SquadDataset, context_to_pos: dict, candidates, k: int = 10) -> dict:
    """Times one batched search_raw() call (not a per-query loop, same
    reasoning as example 11) under a given `candidates` restriction, and
    scores quality only over queries whose correct answer is in scope."""
    index.search_raw(dataset.queries[:1], k=k, parallel=True, candidates=candidates)  # warm-up

    n = len(dataset.queries)
    t0 = time.perf_counter()
    ids, _scores = index.search_raw(dataset.queries, k=k, parallel=True, candidates=candidates)
    query_ms = (time.perf_counter() - t0) * 1000 / n

    candidate_set = set(int(p) for p in candidates) if candidates is not None else None

    in_scope = hit1 = hitk = mrr = 0
    for q, row in zip(dataset.queries, ids):
        correct = dataset.query_to_context.get(q)
        correct_pos = context_to_pos.get(correct)
        if candidate_set is not None and correct_pos not in candidate_set:
            continue  # correct answer was filtered out -- not this filter's fault to explain
        in_scope += 1
        rank = next((i for i, pos in enumerate(row) if pos >= 0 and dataset.corpus[pos] == correct), None)
        if rank is not None:
            if rank == 0:
                hit1 += 1
            hitk += 1
            mrr += 1 / (rank + 1)

    denom = in_scope or 1
    return {
        "name": name,
        "query_ms": query_ms,
        "in_scope": in_scope / n,
        "hit1": hit1 / denom,
        "hitk": hitk / denom,
        "mrr": mrr / denom,
    }


def main() -> None:
    print("Loading dataset...")
    dataset = SquadDataset.from_hf(n_contexts=2000)
    print(f"Dataset: {len(dataset.corpus)} documents & {len(dataset.queries)} queries\n")
    context_to_pos = {ctx: i for i, ctx in enumerate(dataset.corpus)}
    categories = [i % N_CATEGORIES for i in range(len(dataset.corpus))]

    # Auto-generated str(i) ids, matching what every backend's fit() assigns
    # to its own positions -- built independently, no pairing needed.
    sql_filter = SQLFilter()
    sql_filter.add([str(i) for i in range(len(dataset.corpus))], [{"category": c} for c in categories])

    bm25x_index = BM25xIndex()
    t0 = time.perf_counter()
    bm25x_index.fit(dataset.corpus)
    bm25x_build_ms = (time.perf_counter() - t0) * 1000

    bm25c_index = BM25CIndex()
    t0 = time.perf_counter()
    bm25c_index.fit(dataset.corpus)
    bm25c_build_ms = (time.perf_counter() - t0) * 1000

    lookup_index = LookupIndex()
    t0 = time.perf_counter()
    lookup_index.fit(dataset.corpus)
    lookup_build_ms = (time.perf_counter() - t0) * 1000

    relevance_index = RelevanceIndex()
    t0 = time.perf_counter()
    relevance_index.fit(dataset.corpus)
    relevance_build_ms = (time.perf_counter() - t0) * 1000

    print(
        f"Build: bm25x={bm25x_build_ms:.0f}ms  bm25c={bm25c_build_ms:.0f}ms "
        f"lookup={lookup_build_ms:.0f}ms  relevance={relevance_build_ms:.0f}ms\n"
    )

    scenarios = [
        ("no filter (candidates=None)", None),
        ("1/10 of corpus (category = 0)", sql_filter.where("category", "=", 0).positions()),
        ("half of corpus (category < 5)", sql_filter.where("category", "<", 5).positions()),
        ("full corpus (category >= 0)", sql_filter.where("category", ">=", 0).positions()),
    ]

    header = f"{'Backend':<10}{'Filter':<32}{'Query(ms)':>10}{'InScope':>9}{'Hit@1':>8}{'Hit@10':>8}{'MRR':>7}"
    print(header)
    print("-" * len(header))
    backends = [
        ("bm25x", bm25x_index),
        ("bm25c", bm25c_index),
        ("lookup", lookup_index),
        ("relevance", relevance_index),
    ]
    for backend_name, index in backends:
        for scenario_name, candidates in scenarios:
            r = benchmark_scenario(scenario_name, index, dataset, context_to_pos, candidates)
            print(
                f"{backend_name:<10}{r['name']:<32}{r['query_ms']:>10.3f}"
                f"{r['in_scope']:>9.1%}{r['hit1']:>8.1%}{r['hitk']:>8.1%}{r['mrr']:>7.3f}"
            )
        print()


if __name__ == "__main__":
    main()
