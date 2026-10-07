"""Seed HotpotQA (multi-hop QA) and measure authorization under partial
context access: each question needs 2 supporting paragraphs, so a user can
have access to 0, 1 (either one), or both of them.
"""

import random
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_DSN = "host=localhost port=5433 dbname=rags_authorization user=<DB_USER> password=<DB_PASSWORD>"

NUM_QUESTIONS = 90447  # full train split
NUM_ROLES = 10
NUM_USERS = 100
NUM_WORKERS = 8

model = SentenceTransformer("all-MiniLM-L6-v2")


def seed_scale() -> None:
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    data = data.select(range(NUM_QUESTIONS))

    role_names = [f"role_{i}" for i in range(NUM_ROLES)]

    doc_ids: list[str] = []
    doc_titles: list[str] = []
    doc_contents: list[str] = []
    doc_roles: list[str] = []

    for row in data:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        for title, sents in zip(titles, sentences):
            doc_ids.append(str(uuid.uuid4()))
            doc_titles.append(title)
            doc_contents.append(" ".join(sents)[:5000])
            doc_roles.append(random.choice(role_names))

    print(f"Loaded {NUM_QUESTIONS} questions -> {len(doc_ids)} paragraphs.")
    print("Generating embeddings...")
    embeddings = model.encode(doc_contents, batch_size=64, show_progress_bar=True)

    with psycopg.connect(DB_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            role_ids = {}
            for name in role_names:
                cur.execute(
                    "INSERT INTO roles (name) VALUES (%s) "
                    "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
                    "RETURNING id",
                    (name,),
                )
                role_ids[name] = cur.fetchone()[0]

            for i in range(NUM_USERS):
                cur.execute(
                    "INSERT INTO users (email) VALUES (%s) "
                    "ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email "
                    "RETURNING id",
                    (f"probe_{i}@example.com",),
                )
                user_id = cur.fetchone()[0]
                role = random.choice(role_names)
                cur.execute(
                    "INSERT INTO user_roles (user_id, role_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (user_id, role_ids[role]),
                )

            print("Bulk inserting documents...")
            with cur.copy("COPY documents (id, title) FROM STDIN") as copy:
                for doc_id, title in zip(doc_ids, doc_titles):
                    copy.write_row((doc_id, title))

            print("Bulk inserting document_access...")
            with cur.copy("COPY document_access (document_id, role_id) FROM STDIN") as copy:
                for doc_id, role in zip(doc_ids, doc_roles):
                    copy.write_row((doc_id, str(role_ids[role])))

            print("Bulk inserting document_sections...")
            with cur.copy(
                "COPY document_sections (document_id, content, embedding) FROM STDIN"
            ) as copy:
                for doc_id, content, embedding in zip(doc_ids, doc_contents, embeddings):
                    copy.write_row((doc_id, content, str(embedding.tolist())))

    print(f"Scale seed complete: {len(doc_ids)} paragraphs.")


def _role_sets(cur) -> dict:
    cur.execute("SELECT user_id, role_id FROM user_roles")
    roles: dict = {}
    for user_id, role_id in cur.fetchall():
        roles.setdefault(str(user_id), set()).add(str(role_id))
    return roles


def _find_user(role_sets: dict, all_users: list, role_a: str, role_b: str, scenario: str):
    for user_id in all_users:
        has_a = role_a in role_sets.get(user_id, set())
        has_b = role_b in role_sets.get(user_id, set())
        if scenario == "none" and not has_a and not has_b:
            return user_id
        if scenario == "a_only" and has_a and not has_b:
            return user_id
        if scenario == "b_only" and has_b and not has_a:
            return user_id
        if scenario == "both" and has_a and has_b:
            return user_id
    return None


def _lookup_gold_docs(cur, titles: list[str]) -> dict:
    """document_id + role_id for each title, looked up live from the DB."""
    cur.execute(
        "SELECT DISTINCT ON (d.title) d.title, d.id, da.role_id "
        "FROM documents d JOIN document_access da ON da.document_id = d.id "
        "WHERE d.title = ANY(%s)",
        (titles,),
    )
    return {title: (str(doc_id), str(role_id)) for title, doc_id, role_id in cur.fetchall()}


def _diagnose_miss(cur, user_id: str, embedding: str, missing_doc_id: str) -> tuple | None:
    """Rank/distance of an authorized document that fell outside the top-k."""
    cur.execute(
        """
        WITH authorized_sections AS MATERIALIZED (
          SELECT ds.document_id, ds.embedding
          FROM document_sections ds
          WHERE ds.document_id IN (
            SELECT da.document_id FROM document_access da
            JOIN user_roles ur ON ur.role_id = da.role_id
            WHERE ur.user_id = %s
          )
        ), ranked AS (
          SELECT document_id,
                 embedding <=> %s::vector AS distance,
                 row_number() OVER (ORDER BY embedding <=> %s::vector) AS rank
          FROM authorized_sections
        )
        SELECT distance, rank FROM ranked WHERE document_id = %s
        """,
        (user_id, embedding, embedding, missing_doc_id),
    )
    return cur.fetchone()


def _process_question(q: dict, role_sets: dict, all_users: list, top_k: int) -> dict:
    """Runs every applicable scenario for one question on its own connection."""
    result = {
        "leaks": 0,
        "single_recall_hits": 0,
        "single_recall_checks": 0,
        "both_found": None,
        "miss_ranks": [],
        "miss_distances": [],
        "role_clash": q["role_a"] == q["role_b"],
    }
    scenarios = (
        ("none", "both") if result["role_clash"] else ("none", "a_only", "b_only", "both")
    )

    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            for scenario in scenarios:
                user_id = _find_user(role_sets, all_users, q["role_a"], q["role_b"], scenario)
                if user_id is None:
                    continue

                cur.execute(
                    "SELECT document_id FROM search_document_sections(%s, %s::vector, %s)",
                    (user_id, q["embedding"], top_k),
                )
                found_ids = {str(row[0]) for row in cur.fetchall()}
                found_a = q["doc_a"] in found_ids
                found_b = q["doc_b"] in found_ids

                if scenario == "none":
                    if found_a or found_b:
                        result["leaks"] += 1
                elif scenario == "a_only":
                    if found_b:
                        result["leaks"] += 1
                    result["single_recall_checks"] += 1
                    if found_a:
                        result["single_recall_hits"] += 1
                    else:
                        diag = _diagnose_miss(cur, user_id, q["embedding"], q["doc_a"])
                        if diag:
                            result["miss_distances"].append(diag[0])
                            result["miss_ranks"].append(diag[1])
                elif scenario == "b_only":
                    if found_a:
                        result["leaks"] += 1
                    result["single_recall_checks"] += 1
                    if found_b:
                        result["single_recall_hits"] += 1
                    else:
                        diag = _diagnose_miss(cur, user_id, q["embedding"], q["doc_b"])
                        if diag:
                            result["miss_distances"].append(diag[0])
                            result["miss_ranks"].append(diag[1])
                elif scenario == "both":
                    result["both_found"] = int(found_a) + int(found_b)
    return result


def benchmark_authorization(sample_size: int, top_k: int) -> None:
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    indices = random.sample(range(len(data)), min(sample_size, len(data)))

    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            role_sets = _role_sets(cur)
    all_users = list(role_sets.keys())

    questions = []
    skipped_not_found = 0
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            for idx in indices:
                row = data[idx]
                gold_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
                if len(gold_titles) != 2:
                    skipped_not_found += 1
                    continue

                lookup = _lookup_gold_docs(cur, gold_titles)
                if len(lookup) != 2:
                    skipped_not_found += 1
                    continue

                (doc_a, role_a), (doc_b, role_b) = (lookup[t] for t in gold_titles)
                questions.append(
                    {
                        "question": row["question"],
                        "doc_a": doc_a,
                        "role_a": role_a,
                        "doc_b": doc_b,
                        "role_b": role_b,
                    }
                )

    embeddings = model.encode([q["question"] for q in questions], batch_size=64)
    for q, embedding in zip(questions, embeddings):
        q["embedding"] = str(embedding.tolist())

    leaks = 0
    single_recall_hits = 0
    single_recall_checks = 0
    both_found_counts = []
    miss_ranks = []
    miss_distances = []
    role_clashes = 0

    bench_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(_process_question, q, role_sets, all_users, top_k)
            for q in questions
        ]
        for future in tqdm(as_completed(futures), total=len(questions)):
            r = future.result()
            leaks += r["leaks"]
            single_recall_hits += r["single_recall_hits"]
            single_recall_checks += r["single_recall_checks"]
            if r["both_found"] is not None:
                both_found_counts.append(r["both_found"])
            miss_ranks.extend(r["miss_ranks"])
            miss_distances.extend(r["miss_distances"])
            if r["role_clash"]:
                role_clashes += 1

    total_elapsed = time.perf_counter() - bench_start

    recall_1 = 100 * single_recall_hits / max(single_recall_checks, 1)
    recall_2 = 100 * both_found_counts.count(2) / max(len(both_found_counts), 1)
    recall_2_partial = 100 * sum(1 for c in both_found_counts if c >= 1) / max(len(both_found_counts), 1)
    avg_time = total_elapsed / max(len(questions), 1)

    print()
    print(f"Questions: {len(questions)}")
    print(f"Leaked: {leaks}")
    print(f"Recall 1: {recall_1:.1f}%")
    print(f"Recall 2 (1/2): {recall_2_partial:.1f}%")
    print(f"Recall 2: {recall_2:.1f}%")
    print(f"Avg time: {avg_time * 1000:.1f}ms")
    print(f"Total time: {total_elapsed / 60:.1f}min")


if __name__ == "__main__":
    if "seed" in sys.argv:
        seed_scale()
    benchmark_authorization(sample_size=10_000, top_k=100)
