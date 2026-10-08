#!/usr/bin/env python

# %%
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from simlar import CallableEmbedder, FilteredIndex, FilterError, HelixIndex, load_from_directory

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "pgvector_recipe_demo" / "data" / "recipes_sample.csv"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 5

LIST_COLUMNS = {
    "cuisine_list": "cuisines",
    "course_list": "courses",
    "tastes": "tastes",
    "dietary_profile": "diets",
    "health_flags": "health_flags",
}


# ── Data ──────────────────────────────────────────────────────────────────────


def load_recipes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for src, dst in LIST_COLUMNS.items():
        df[dst] = df[src].apply(json.loads)
    df["prep_min"] = df["est_prep_time_min"]
    df["cook_min"] = df["est_cook_time_min"]
    df["total_min"] = df["prep_min"] + df["cook_min"]
    df = df.reset_index(drop=True)
    df["id"] = [f"r{i}" for i in range(len(df))]
    return df


def recipe_metadata(row) -> dict:
    """One recipe's filterable metadata: numbers, categories and lists -- no
    booleans (dietary flags are the `diets` list instead)."""
    return {
        # numeric
        "total_min": int(row.total_min),
        "prep_min": int(row.prep_min),
        "cook_min": int(row.cook_min),
        "healthiness_score": int(row.healthiness_score),
        "num_ingredients": int(row.num_ingredients),
        "num_steps": int(row.num_steps),
        # categorical
        "difficulty": row.difficulty,
        "category": row.category,
        "primary_taste": row.primary_taste,
        "cook_speed": row.cook_speed,
        "main_ingredient": row.main_ingredient,
        "health_level": row.health_level,
        # multi-valued
        "cuisines": list(row.cuisines),
        "courses": list(row.courses),
        "tastes": list(row.tastes),
        "diets": list(row.diets),
        "health_flags": list(row.health_flags),
    }


def recipe_roles(row) -> list[str]:
    """Toy access model: 'hard' recipes are premium-only, the rest are
    visible to both tiers."""
    return ["premium"] if row.difficulty == "hard" else ["free", "premium"]


# ── Output + verification ─────────────────────────────────────────────────────


def show(title: str, df: pd.DataFrame, results, check=None, extra=()) -> None:
    """Print results and assert each one satisfies `check(row)`."""
    print(f"\n── {title}")
    if not results:
        print("   (no results)")
    cols = ["difficulty", "total_min", "healthiness_score", *extra]
    for r in results:
        row = df.loc[int(r.id[1:])]
        info = ", ".join(f"{c}={_fmt(row[c])}" for c in cols)
        print(f"   {r.score:7.4f}  {row.recipe_title[:48]:<48}  {info}")
        if check is not None:
            assert check(row), f"filter violated by {r.id}: {row.recipe_title!r}"


def _fmt(value) -> str:
    return "/".join(value) if isinstance(value, list) else str(value)


def count(idx: FilteredIndex, filter=None, roles=None) -> int:
    return len(idx.candidates(filter, roles))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = load_recipes(csv_path)
    print(f"Loaded {len(df)} recipes from {csv_path}")

    from sentence_transformers import SentenceTransformer

    # The embedder turns recipe texts into vectors on add() and query text into
    # a vector on search(). text_k / vector_k are left unset: the engine tunes
    # them to the corpus size and top_k.
    embedder = CallableEmbedder(SentenceTransformer(MODEL).encode, normalize=True, batch_size=64)
    texts = df["combined_text"].fillna("").tolist()

    t0 = time.perf_counter()
    idx = FilteredIndex(HelixIndex(embedder=embedder))
    idx.add(
        df["id"].tolist(),
        texts,
        metadata=[recipe_metadata(row) for row in df.itertuples()],
        roles=[recipe_roles(row) for row in df.itertuples()],
    )
    scalar_cols, multi_keys = idx.sql_filter.schema()
    print(f"Embedded with {MODEL} and built FilteredIndex in {time.perf_counter() - t0:.1f}s "
          f"over {idx.size} recipes")
    print(f"  speed_preference: {idx.speed_preference}  expected recall: {idx.expected_recall}")
    print(f"  scalar columns: {sorted(scalar_cols)}")
    print(f"  multi-valued:   {sorted(multi_keys)}")

    # Vector-only searches (as in LangChain's similarity_search_by_vector) take
    # a query vector; hybrid searches take just the text.
    embed = embedder.embed_query

    # ── 1. Vector-only search, numeric filter ──
    # The shape of LangChain's similarity_search_by_vector(query_vector, filter=...).
    q_text = "quick weeknight chicken dinner"
    q = embed(q_text)
    f = "total_min <= 30"
    print(f"\n[1] vector-only, filter={f!r}  ({count(idx, f)} recipes qualify)")
    show("unfiltered", df, idx.search(query_vector=q, k=K))
    show(f"filter: {f}", df, idx.search(query_vector=q, k=K, filter=f),
         check=lambda r: r.total_min <= 30)

    # ── 2. Hybrid search, multi-valued + numeric ──
    f = "'dessert' IN courses AND healthiness_score >= 70"
    print(f"\n[2] hybrid 'chocolate cake', filter={f!r}  ({count(idx, f)} qualify)")
    show(f"filter: {f}", df,
         idx.search(query_text="chocolate cake", k=K, filter=f),
         check=lambda r: "dessert" in r.courses and r.healthiness_score >= 70, extra=("courses",))

    # ── 3. Categories, IN, BETWEEN ──
    f = "difficulty IN ('easy', 'medium') AND num_ingredients BETWEEN 3 AND 8 AND cook_speed = 'fast'"
    print(f"\n[3] hybrid 'pasta', filter={f!r}  ({count(idx, f)} qualify)")
    show(f"filter: {f}", df,
         idx.search(query_text="pasta", k=K, filter=f),
         check=lambda r: r.difficulty in ("easy", "medium") and 3 <= r.num_ingredients <= 8
         and r.cook_speed == "fast", extra=("num_ingredients", "cook_speed"))

    # ── 4. Several list columns, OR, NOT ──
    f = ("'vegan' IN diets AND ('spicy' IN tastes OR primary_taste = 'savory') "
         "AND NOT 'fried' IN health_flags")
    q_text = "hearty stew"
    q = embed(q_text)
    print(f"\n[4] vector 'hearty stew', filter={f!r}  ({count(idx, f)} qualify)")
    show(f"filter: {f}", df, idx.search(query_vector=q, k=K, filter=f),
         check=lambda r: "vegan" in r.diets and ("spicy" in r.tastes or r.primary_taste == "savory")
         and "fried" not in r.health_flags, extra=("diets", "tastes"))

    # ── 5. LIKE, and the same filter as a MetadataFilter chain ──
    # SQLite LIKE is ASCII case-insensitive. Strings and dicts are translated
    # into a MetadataFilter, so a chain built by hand selects the same rows.
    f = "category LIKE '%Bread%' AND health_level NOT LIKE 'unhealthy' AND prep_min <= 20"
    sf = idx.sql_filter
    chain = sf.where_like("category", "%Bread%").where("prep_min", "<=", 20) & ~sf.where_like(
        "health_level", "unhealthy"
    )
    assert idx.candidates(chain).tolist() == idx.candidates(f).tolist()
    print(f"\n[5] hybrid 'banana bread', filter={f!r}  ({count(idx, f)} qualify)")
    show(f"filter: {f}", df,
         idx.search(query_text="banana bread", k=K, filter=chain),
         check=lambda r: "bread" in r.category.lower() and r.health_level.lower() != "unhealthy"
         and r.prep_min <= 20, extra=("category", "prep_min", "health_level"))

    # ── 6. The same filter as a LangChain-style dict ──
    d = {"$and": [
        {"courses": {"$contains": "soup"}},
        {"cuisines": {"$in": ["asian", "korean", "thai"]}},
        {"total_min": {"$lte": 60}},
    ]}
    s = "'soup' IN courses AND cuisines IN ('asian', 'korean', 'thai') AND total_min <= 60"
    assert idx.candidates(d).tolist() == idx.candidates(s).tolist()
    print(f"\n[6] dict filter == SQL filter ({count(idx, d)} qualify)")
    show("dict: soup, asian/korean/thai, <= 60 min", df,
         idx.search(query_vector=embed("noodle soup"), k=K, filter=d),
         check=lambda r: "soup" in r.courses and bool({"asian", "korean", "thai"} & set(r.cuisines))
         and r.total_min <= 60, extra=("cuisines",))

    # ── 7. Roles ──
    q = embed("slow braised beef")
    print(f"\n[7] roles: free sees {count(idx, roles=['free'])}, "
          f"premium sees {count(idx, roles=['premium'])} of {idx.size}")
    show("roles=['free']  (no 'hard' recipes)", df, idx.search(query_vector=q, k=K, roles=["free"]),
         check=lambda r: r.difficulty != "hard")
    show("roles=['premium']", df, idx.search(query_vector=q, k=K, roles=["premium"]))
    # A filter can't widen access: the role check is appended by FilteredIndex.
    show("roles=['free'], filter='1=1 OR 1=1'", df,
         idx.search(query_vector=q, k=K, roles=["free"], filter="1=1 OR 1=1"),
         check=lambda r: r.difficulty != "hard")

    # ── 8. Pre-filter vs post-filter ──
    # A selective filter applied *after* an unrestricted top-k under-returns;
    # FilteredIndex applies it before ranking.
    f = "'korean' IN cuisines AND 'breakfast' IN courses AND total_min <= 45"
    q = embed("pancakes")
    allowed = {f"r{p}" for p in idx.candidates(f)}
    post = [r for r in idx.search(query_vector=q, k=K) if r.id in allowed]
    pre = idx.search(query_vector=q, k=K, filter=f)
    print(f"\n[8] {f!r}: {len(allowed)} qualify; "
          f"post-filtering the unfiltered top-{K} keeps {len(post)}, pre-filtering returns {len(pre)}")
    assert len(pre) == min(K, len(allowed))

    # ── 9. Rejected filters ──
    print("\n[9] rejected filters")
    for bad in [
        "total_min <= 30; DROP TABLE docs",
        "total_min IN (SELECT position FROM doc_access)",
        "load_extension('evil.so')",
        "calories < 500",                       # unknown column
        "healthiness_score > 50 AND tastes > 'a'",  # range on a list column
    ]:
        try:
            idx.search(query_vector=q, k=K, filter=bad)
            raise AssertionError(f"accepted: {bad}")
        except FilterError as e:
            print(f"   {bad!r:<52} -> {str(e)[:70]}")

    # ── 10. Metadata updates + persistence ──
    first = idx.search(query_vector=embed("pumpkin"), k=1)[0].id
    idx.update_metadata([first], [{"healthiness_score": 999}])
    assert [r.id for r in idx.search(query_vector=q, k=K, filter="healthiness_score = 999")] == [first]
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "recipes")
        idx.save(path)
        loaded = load_from_directory(path, embedder=embedder)  # embedders aren't saved
        f = "'dessert' IN courses AND total_min <= 30"
        a = [r.id for r in idx.search(query_vector=q, k=K, filter=f, roles=["free"])]
        b = [r.id for r in loaded.search(query_vector=q, k=K, filter=f, roles=["free"])]
        assert type(loaded) is FilteredIndex and a == b
    print(f"\n[10] update_metadata + save/load round trip OK ({type(loaded.inner).__name__} inner)")

    print("\nAll filtered results verified against the source data.")


if __name__ == "__main__":
    main()

# %%
