#!/usr/bin/env python
"""SQLFilter -- pre-filter a corpus before search runs, over real arXiv metadata.

SQLFilter (simlar_engine._sql_filter.SQLFilter) is a standalone SQL-driven
metadata/access-control pre-filter: it owns its own SQLite connection and
its own id<->position table, assigning positions by strict append order on
add() -- the same alignment contract every TextIndex/VectorIndex backend
follows. Building a SQLFilter with the same ids in the same order as a
sibling TextIndex's fit()/add() keeps the two positionally aligned without
either needing to know about the other.

It answers three different kinds of questions:

  - `where(col, op, val)...`     -- a fluent, AND-chained condition builder
                                    over per-doc *scalar* metadata (one
                                    value per doc, e.g. a paper's primary
                                    category), terminated by `.positions()`.
  - `where_any(col, op, val)...` -- the same fluent chain, but over a
                                    *multi-valued* attribute (a paper can
                                    have several arXiv categories at once)
                                    -- true if ANY of the doc's values for
                                    `col` satisfies `op val`.
  - `filter_by_roles(roles)`     -- which docs a given role may access at
                                    all (permissions, not a property of the
                                    doc's content -- a doc doesn't "have" a
                                    role).

None of these ever see raw SQL text from the caller -- column names are
validated as identifiers and operators against a fixed whitelist, so the
only caller-controlled thing that reaches the database is `value`, always
as a bound parameter.

Either `where()`/`where_any()`'s `.positions()` or `filter_by_roles()`
returns a plain array of positions, which any TextIndex backend's
`search`/`search_raw` `candidates` parameter uses to prune the corpus
*before* scoring -- not a post-hoc filter of an already-ranked top-k, which
would under-return whenever the true top-k-within-candidates isn't among the
top-k-unrestricted.

Corpus: the first N papers (with a usable abstract and categories) streamed
from the arXiv metadata snapshot on HuggingFace -- a flat, one-JSON-object-
per-paper schema. `categories` is genuinely multi-valued (a paper can be
cross-listed under several, e.g. "math.CO cs.CG"), which is exactly the
shape `where_any()` exists for; arXiv has no real access-control data, so
the role demo below repurposes a real field (whether a paper has a
`journal-ref`, i.e. made it to a peer-reviewed publication) as a stand-in
for "who can see this doc", the same kind of illustrative scenario the
original version of this example used with invented department roles.
"""

from __future__ import annotations

from datasets import load_dataset
from simlar import BM25CIndex
from simlar_engine import SQLFilter

N_PAPERS = 300


def _load_papers(n: int) -> list[dict]:
    ds = load_dataset("jackkuo/arXiv-metadata-oai-snapshot", split="train", streaming=True)
    papers = []
    for row in ds:
        if not row.get("abstract") or not row.get("categories"):
            continue
        papers.append(row)
        if len(papers) >= n:
            break
    return papers


def main() -> None:
    print(f"Streaming the first {N_PAPERS} arXiv papers with an abstract + categories...")
    papers = _load_papers(N_PAPERS)

    ids = [p["id"] for p in papers]
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    categories = [p["categories"].split() for p in papers]  # multi-valued
    primary_category = [c[0] for c in categories]  # scalar: first-listed category
    update_date = [p["update_date"].strftime("%Y-%m-%d") for p in papers]  # ISO -> plain string compare
    published = [bool(p["journal-ref"]) for p in papers]

    # index and filt are built independently, but on the identical ids in
    # the identical order -- that's the whole alignment contract, no shared
    # connection or pairing object required.
    index = BM25CIndex()
    index.add(ids, texts)

    filt = SQLFilter()
    filt.add(
        ids,
        [
            {"primary_category": pc, "update_date": d, "categories": cs}
            for pc, d, cs in zip(primary_category, update_date, categories, strict=True)
        ],
    )
    for id_, is_published in zip(ids, published, strict=True):
        filt.grant([id_], ["published"] if is_published else ["preprint"])

    top_category = max(set(primary_category), key=primary_category.count)
    query = "quantum field"

    print(f"Most common primary category in this sample: {top_category!r}\n")
    print(f'Query: "{query}"\n')

    print("-- Unrestricted (candidates=None) --")
    unrestricted = index.search(query, k=5, candidates=None)
    _print(unrestricted)

    print(f"\n-- Scalar filter: primary_category = {top_category!r} (where) --")
    scalar_candidates = filt.where("primary_category", "=", top_category).positions()
    _print_results(index, query, candidates=scalar_candidates)

    # Pick a real cross-listed paper from the unrestricted results themselves:
    # one whose PRIMARY category differs from a category it's ALSO listed
    # under, so where_any() can show something where() genuinely cannot --
    # not a contrived example, an actual paper from the query's own top hits.
    cat_by_id = dict(zip(ids, categories, strict=True))
    cross_listed_secondary = next(
        (c for r in unrestricted for c in cat_by_id[r.id][1:]),
        None,
    )
    print(f"\n-- Multi-valued filter: categories includes {cross_listed_secondary!r} (where_any) --")
    print("   (matches a paper whose PRIMARY category is something else entirely,")
    print("    but that's cross-listed under this one too -- where() alone can't see that)")
    multi_candidates = filt.where_any("categories", "=", cross_listed_secondary).positions()
    _print_results(index, query, candidates=multi_candidates)

    print("\n-- Chained: update_date >= 2008-01-01 AND categories includes 'gr-qc' --")
    chained_candidates = (
        filt.where("update_date", ">=", "2008-01-01").where_any("categories", "=", "gr-qc").positions()
    )
    _print_results(index, query, candidates=chained_candidates)

    print("\n-- Access filter: what a 'published' (has a journal-ref) role can see --")
    published_visible = filt.filter_by_roles(["published"])
    _print_results(index, query, candidates=published_visible)
    print(f"   ({len(ids) - len(published_visible)} of {len(ids)} papers are preprint-only -- excluded)")


def _print_results(index: BM25CIndex, query: str, candidates) -> None:
    _print(index.search(query, k=5, candidates=candidates))


def _print(results) -> None:
    if not results:
        print("  (no matches)")
    for r in results:
        if r.id is not None:
            print(f"  {r.id:14s} score={r.score:.3f}")


if __name__ == "__main__":
    main()
