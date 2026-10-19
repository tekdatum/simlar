"""Seed HotpotQA (multi-hop QA) and measure authorization under partial
context access: each question needs 2 supporting paragraphs, so a user can
have access to 0, 1 (either one), or both of them.

FilteredIndex variant of hotpotqa.py: same flow, scenarios and metrics, with
authorization handled by simlar's FilteredIndex instead of a hand-paired
SimlarEngine + SQLFilter:

  - document_sections.embedding + HNSW  -> SimlarEngine, inside FilteredIndex
  - document_access / roles             -> FilteredIndex.add(roles=...)
  - users / user_roles                  -> in-memory dict (user -> role names)
  - search_document_sections(user, ...) -> FilteredIndex.search_raw(roles=...)

Everything is built in memory on each run; nothing is persisted. Paragraph
text is only used for the embeddings and then discarded.

FilteredIndex resolves roles through a SQLite connection, which can't cross
threads. Every role set's candidates are resolved once in the main thread,
which fills FilteredIndex's cache; the workers' searches then only read it.

Searches that share a role set are split into chunks, and NUM_WORKERS
threads each run one chunk per search_raw call.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from simlar import FilteredIndex, SimlarEngine
from tqdm import tqdm

NUM_QUESTIONS = 90447  # full train split
NUM_ROLES = 10
NUM_USERS = 100
NUM_WORKERS = 8
BATCH_SIZE = 64  # queries per search_raw call (one chunk per worker task)
DIAG_BATCH_SIZE = 32  # full-ranking diagnosis returns every candidate per query

model = SentenceTransformer("all-MiniLM-L6-v2")


def _doc_id(question_idx: int, paragraph_idx: int) -> str:
    """Deterministic paragraph id, so gold paragraphs resolve without a title lookup."""
    return f"{question_idx}:{paragraph_idx}"


def build_index() -> tuple[FilteredIndex, dict]:
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    data = data.select(range(NUM_QUESTIONS))

    role_names = [f"role_{i}" for i in range(NUM_ROLES)]

    doc_ids: list[str] = []
    doc_contents: list[str] = []
    doc_roles: list[list[str]] = []

    for question_idx, row in enumerate(data):
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        for paragraph_idx, (title, sents) in enumerate(zip(titles, sentences)):
            doc_ids.append(_doc_id(question_idx, paragraph_idx))
            doc_contents.append(" ".join(sents)[:5000])
            doc_roles.append([random.choice(role_names)])

    print(f"Loaded {NUM_QUESTIONS} questions -> {len(doc_ids)} paragraphs.")
    print("Generating embeddings...")
    embeddings = model.encode(doc_contents, batch_size=64, show_progress_bar=True)
    del doc_contents

    users = {f"probe_{i}@example.com": [random.choice(role_names)] for i in range(NUM_USERS)}

    print("Building FilteredIndex (SimlarEngine + roles)...")
    index = FilteredIndex(SimlarEngine())
    index.add(doc_ids, np.asarray(embeddings, dtype=np.float32), roles=doc_roles, parallel=True)

    print(f"Index build complete: {len(doc_ids)} paragraphs.")
    role_sets = {user_id: set(roles) for user_id, roles in users.items()}
    return index, role_sets


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


def _lookup_gold_docs(
    row: dict, question_idx: int, titles: list[str], pos_of_doc: dict, role_of_pos: np.ndarray
) -> dict:
    """document_id + role for each title, from this question's own paragraphs."""
    context_titles = row["context"]["title"]
    lookup = {}
    for title in titles:
        if title not in context_titles:
            continue
        doc_id = _doc_id(question_idx, context_titles.index(title))
        pos = pos_of_doc.get(doc_id)
        if pos is not None and role_of_pos[pos] is not None:
            lookup[title] = (doc_id, role_of_pos[pos])
    return lookup


def _batched_search(
    index: FilteredIndex,
    embeddings: np.ndarray,
    roles: tuple,
    k: int,
    batch_size: int,
    n_candidates: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector search of many queries restricted to what `roles` may see."""
    ranked_pos, ranked_dist = [], []
    for start in range(0, len(embeddings), batch_size):
        pos, dist = index.search_raw(
            embeddings[start : start + batch_size],
            k=k,
            roles=list(roles),
            parallel=True,
            n_candidates=n_candidates,
        )
        ranked_pos.append(pos)
        ranked_dist.append(dist)
    return np.concatenate(ranked_pos), np.concatenate(ranked_dist)


def _diagnose_misses(
    index: FilteredIndex, misses: list[tuple[np.ndarray, int]], roles: tuple, n_allowed: int
) -> list[tuple[float, int]]:
    """Rank/distance of each authorized document that fell outside the top-k."""
    embeddings = np.stack([embedding for embedding, _ in misses])
    ranked_pos, ranked_dist = _batched_search(
        index, embeddings, roles, n_allowed, DIAG_BATCH_SIZE, n_candidates=n_allowed
    )
    diagnoses = []
    for (_, missing_pos), row_pos, row_dist in zip(misses, ranked_pos, ranked_dist):
        hits = np.flatnonzero(row_pos == missing_pos)
        if hits.size:
            rank = int(hits[0])
            diagnoses.append((float(row_dist[rank]), rank + 1))
    return diagnoses


def _process_chunk(
    tasks: list[tuple[dict, str]],
    roles: tuple,
    candidates: np.ndarray,
    index: FilteredIndex,
    doc_of_pos: np.ndarray,
    top_k: int,
) -> dict:
    """Runs one chunk of (question, scenario) searches that share a role set."""
    result = {
        "leaks": 0,
        "out_of_candidates": 0,
        "single_recall_hits": 0,
        "single_recall_checks": 0,
        "both_found": [],
        "miss_ranks": [],
        "miss_distances": [],
    }

    if candidates.size == 0:
        ranked = [np.empty(0, dtype=np.int64)] * len(tasks)
    else:
        embeddings = np.stack([q["embedding"] for q, _ in tasks])
        ranked, _ = _batched_search(index, embeddings, roles, min(top_k, candidates.size), BATCH_SIZE)

    misses = []
    for (q, scenario), ranked_pos in zip(tasks, ranked):
        ranked_pos = ranked_pos[ranked_pos >= 0]  # -1 pads rows with fewer allowed docs than k
        result["out_of_candidates"] += int((~np.isin(ranked_pos, candidates)).sum())
        found_ids = set(doc_of_pos[ranked_pos])
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
                misses.append((q["embedding"], q["pos_a"]))
        elif scenario == "b_only":
            if found_a:
                result["leaks"] += 1
            result["single_recall_checks"] += 1
            if found_b:
                result["single_recall_hits"] += 1
            else:
                misses.append((q["embedding"], q["pos_b"]))
        elif scenario == "both":
            result["both_found"].append(int(found_a) + int(found_b))

    if misses:
        for distance, rank in _diagnose_misses(index, misses, roles, candidates.size):
            result["miss_distances"].append(distance)
            result["miss_ranks"].append(rank)
    return result


def benchmark_authorization(sample_size: int, top_k: int) -> None:
    index, role_sets = build_index()
    doc_of_pos = np.array(index.ids, dtype=object)
    pos_of_doc = {doc_id: pos for pos, doc_id in enumerate(index.ids)}
    all_users = list(role_sets.keys())

    # Resolved in the main thread: fills FilteredIndex's per-roles cache, so
    # the workers' search_raw(roles=...) calls never touch SQLite. Also kept
    # here to check every result against what the user may see.
    candidates_by_roles = {
        roles: index.candidates(roles=list(roles))
        for roles in {tuple(sorted(r)) for r in role_sets.values()}
    }

    role_of_pos = np.full(len(doc_of_pos), None, dtype=object)
    for role in sorted(set().union(*role_sets.values())):
        role_of_pos[index.candidates(roles=[role])] = role

    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    indices = random.sample(range(len(data)), min(sample_size, len(data)))

    questions = []
    skipped_not_found = 0
    for idx in indices:
        row = data[idx]
        gold_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
        if len(gold_titles) != 2:
            skipped_not_found += 1
            continue

        lookup = _lookup_gold_docs(row, idx, gold_titles, pos_of_doc, role_of_pos)
        if len(lookup) != 2:
            skipped_not_found += 1
            continue

        (doc_a, role_a), (doc_b, role_b) = (lookup[t] for t in gold_titles)
        questions.append(
            {
                "question": row["question"],
                "doc_a": doc_a,
                "pos_a": pos_of_doc[doc_a],
                "role_a": role_a,
                "doc_b": doc_b,
                "pos_b": pos_of_doc[doc_b],
                "role_b": role_b,
            }
        )

    embeddings = model.encode([q["question"] for q in questions], batch_size=64)
    for q, embedding in zip(questions, embeddings):
        q["embedding"] = np.asarray(embedding, dtype=np.float32)

    # (question, scenario) for every applicable scenario, grouped by the
    # user's role set so each group shares one candidate set.
    tasks_by_roles: dict[tuple, list] = {}
    for q in questions:
        role_clash = q["role_a"] == q["role_b"]
        scenarios = ("none", "both") if role_clash else ("none", "a_only", "b_only", "both")
        for scenario in scenarios:
            user_id = _find_user(role_sets, all_users, q["role_a"], q["role_b"], scenario)
            if user_id is None:
                continue
            roles = tuple(sorted(role_sets[user_id]))
            tasks_by_roles.setdefault(roles, []).append((q, scenario))

    # One worker task per chunk of BATCH_SIZE searches sharing a role set.
    chunks = [
        (tasks[start : start + BATCH_SIZE], roles, candidates_by_roles[roles])
        for roles, tasks in tasks_by_roles.items()
        for start in range(0, len(tasks), BATCH_SIZE)
    ]

    leaks = 0
    out_of_candidates = 0
    single_recall_hits = 0
    single_recall_checks = 0
    both_found_counts = []
    miss_ranks = []
    miss_distances = []

    bench_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(_process_chunk, tasks, roles, candidates, index, doc_of_pos, top_k)
            for tasks, roles, candidates in chunks
        ]
        for future in tqdm(as_completed(futures), total=len(chunks)):
            r = future.result()
            leaks += r["leaks"]
            out_of_candidates += r["out_of_candidates"]
            single_recall_hits += r["single_recall_hits"]
            single_recall_checks += r["single_recall_checks"]
            both_found_counts.extend(r["both_found"])
            miss_ranks.extend(r["miss_ranks"])
            miss_distances.extend(r["miss_distances"])

    total_elapsed = time.perf_counter() - bench_start

    recall_1 = 100 * single_recall_hits / max(single_recall_checks, 1)
    recall_2 = 100 * both_found_counts.count(2) / max(len(both_found_counts), 1)
    recall_2_partial = 100 * sum(1 for c in both_found_counts if c >= 1) / max(len(both_found_counts), 1)
    avg_time = total_elapsed / max(len(questions), 1)

    print()
    print(f"Questions: {len(questions)}")
    print(f"Leaked: {leaks}")
    print(f"Out-of-candidates results: {out_of_candidates}")
    print(f"Recall 1: {recall_1:.1f}%")
    print(f"Recall 2 (1/2): {recall_2_partial:.1f}%")
    print(f"Recall 2: {recall_2:.1f}%")
    print(f"Avg time: {avg_time * 1000:.1f}ms")
    print(f"Total time: {total_elapsed / 60:.1f}min")


if __name__ == "__main__":
    benchmark_authorization(sample_size=10_000, top_k=100)
