#!/usr/bin/env python
"""SQLFilter -- pre-filter a corpus before search runs.

SQLFilter (simlar_engine._sql_filter.SQLFilter) is a standalone SQL-driven
metadata/access-control pre-filter: it owns its own SQLite connection and
its own id<->position table, assigning positions by strict append order on
add() -- the same alignment contract every TextIndex/VectorIndex backend
follows. Building a SQLFilter with the same ids in the same order as a
sibling TextIndex's fit()/add() keeps the two positionally aligned without
either needing to know about the other.

It answers two different kinds of questions:

  - `where(col, op, val)...` -- a fluent, AND-chained condition builder over
                                 per-doc metadata (topical/attribute
                                 filtering, e.g. category), terminated by
                                 `.positions()`. No raw SQL text ever comes
                                 from the caller -- column names and operators
                                 are validated, values are always bound
                                 parameters.
  - `filter_by_roles(roles)` -- which docs a given role may access at all
                                 (permissions, not a property of the doc's
                                 content -- a doc doesn't "have" a role)

Either one returns a plain array of positions, which any TextIndex backend's
`search`/`search_raw` `candidates` parameter uses to prune the corpus
*before* scoring -- not a post-hoc filter of an already-ranked top-k, which
would under-return whenever the true top-k-within-candidates isn't among the
top-k-unrestricted.
"""

from simlar import BM25CIndex
from simlar_engine import SQLFilter

CORPUS = [
    ("eng-1", "The build pipeline runs unit tests on every pull request", "engineering", ["employee"]),
    ("eng-2", "The new caching layer cut API latency by half", "engineering", ["employee"]),
    ("fin-1", "Q3 revenue exceeded projections across all regions", "finance", ["finance-team"]),
    ("fin-2", "The budget review moved the marketing spend forecast up", "finance", ["finance-team"]),
    ("hr-1", "Open enrollment for health benefits starts next week", "hr", ["employee"]),
    ("hr-2", "The new hire onboarding checklist was simplified this quarter", "hr", ["employee"]),
]


def main() -> None:
    ids = [row[0] for row in CORPUS]
    texts = [row[1] for row in CORPUS]
    categories = [row[2] for row in CORPUS]
    roles = [row[3] for row in CORPUS]

    # index and filt are built independently, but on the identical ids in
    # the identical order -- that's the whole alignment contract, no shared
    # connection or pairing object required.
    index = BM25CIndex()
    index.add(ids, texts)

    filt = SQLFilter()
    filt.add(ids, [{"category": c} for c in categories])
    for id_, doc_roles in zip(ids, roles):
        filt.grant([id_], doc_roles)

    query = "the new"

    print(f'Query: "{query}"\n')

    print("-- Unrestricted (candidates=None) --")
    _print_results(index, query, candidates=None)

    print("\n-- Metadata filter: category = 'engineering' --")
    engineering = filt.where("category", "=", "engineering").positions()
    _print_results(index, query, candidates=engineering)

    print("\n-- Access filter: what an 'employee' role can see --")
    employee_visible = filt.filter_by_roles(["employee"])
    _print_results(index, query, candidates=employee_visible)

    # A query that actually hits a finance-team-only doc, to prove the access
    # filter excludes a real match rather than one that just never mattered.
    finance_query = "revenue"
    print(f'\nQuery: "{finance_query}"\n')

    print("-- Unrestricted (candidates=None) --")
    _print_results(index, finance_query, candidates=None)

    print("\n-- Access filter: what an 'employee' role can see --")
    print("   (fin-1 matches 'revenue' but is finance-team-only -- excluded)")
    _print_results(index, finance_query, candidates=employee_visible)


def _print_results(index: BM25CIndex, query: str, candidates) -> None:
    results = index.search(query, k=5, candidates=candidates)
    if not results:
        print("  (no matches)")
    for r in results:
        if r.id is not None:
            print(f"  {r.id:6s} score={r.score:.3f}")


if __name__ == "__main__":
    main()
