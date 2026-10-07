"""Seed HotpotQA (multi-hop QA) and measure authorization under partial
context access: each question needs 2 supporting paragraphs, so a user can
have access to 0, 1 (either one), or both of them.

BM25 variant of hotpotqa_authorization.py, following simlar's SQLFilter
examples (examples/12-14): keyword search with RelevanceIndex and a flat
`candidates=` array from SQLFilter, instead of vector search.

  - document_sections.content           -> RelevanceIndex (BM25)
  - document_access / roles             -> SQLFilter (grant / filter_by_roles)
  - users / user_roles                  -> in-memory dict (user -> role names)
  - search_document_sections(user, ...) -> filter_by_roles(user's roles) as
                                           `candidates=` of a BM25 search

Everything is built in memory on each run; nothing is persisted.

No worker threads: RelevanceIndex tokenizes queries with a PyStemmer
instance, which must not be called concurrently. Searches that share a
candidate set go through batched search_raw calls with parallel=True instead.
"""

import random
import time

import numpy as np
from datasets import load_dataset
from simlar import RelevanceIndex
from simlar_engine import SQLFilter
from tqdm import tqdm

NUM_QUESTIONS = 90447  # full train split
NUM_ROLES = 10
NUM_USERS = 100
BATCH_SIZE = 512  # queries per search_raw call
DIAG_BATCH_SIZE = 32  # full-ranking diagnosis returns every candidate per query


def _doc_id(question_idx: int, paragraph_idx: int) -> str:
    """Deterministic paragraph id, so gold paragraphs resolve without a title lookup."""
    return f"{question_idx}:{paragraph_idx}"


def build_index() -> tuple[RelevanceIndex, SQLFilter, dict]:
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    data = data.select(range(NUM_QUESTIONS))

    role_names = [f"role_{i}" for i in range(NUM_ROLES)]

    doc_ids: list[str] = []
    doc_contents: list[str] = []
    doc_roles: list[str] = []

    for question_idx, row in enumerate(data):
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        for paragraph_idx, (title, sents) in enumerate(zip(titles, sentences)):
            doc_ids.append(_doc_id(question_idx, paragraph_idx))
            doc_contents.append(" ".join(sents)[:5000])
            doc_roles.append(random.choice(role_names))

    print(f"Loaded {NUM_QUESTIONS} questions -> {len(doc_ids)} paragraphs.")

    users = {f"probe_{i}@example.com": [random.choice(role_names)] for i in range(NUM_USERS)}

    # Same ids, same order -> index positions and filter positions line up.
    print("Building RelevanceIndex (BM25)...")
    index = RelevanceIndex()
    index.add(doc_ids, doc_contents)

    print("Building SQLFilter (document_access)...")
    filt = SQLFilter()
    filt.add(doc_ids)
    docs_by_role: dict[str, list[str]] = {}
    for doc_id, role in zip(doc_ids, doc_roles):
        docs_by_role.setdefault(role, []).append(doc_id)
    for role, ids in docs_by_role.items():
        filt.grant(ids, [role])

    print(f"Index build complete: {len(doc_ids)} paragraphs.")
    role_sets = {user_id: set(roles) for user_id, roles in users.items()}
    return index, filt, role_sets


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
    index: RelevanceIndex,
    queries: list[str],
    candidates: np.ndarray,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """BM25 search of many queries restricted to one shared, flat candidate set."""
    ranked_pos, ranked_scores = [], []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start : start + batch_size]
        pos, scores = index.search_raw(chunk, k=k, parallel=True, candidates=candidates)
        ranked_pos.append(pos)
        ranked_scores.append(scores)
    return np.concatenate(ranked_pos), np.concatenate(ranked_scores)


def _diagnose_misses(
    index: RelevanceIndex, misses: list[tuple[str, int]], candidates: np.ndarray
) -> list[tuple[float, int]]:
    """Rank/BM25 score of each authorized document that fell outside the top-k."""
    queries = [question for question, _ in misses]
    ranked_pos, ranked_scores = _batched_search(index, queries, candidates, candidates.size, DIAG_BATCH_SIZE)
    diagnoses = []
    for (_, missing_pos), row_pos, row_scores in zip(misses, ranked_pos, ranked_scores):
        hits = np.flatnonzero(row_pos == missing_pos)
        if hits.size:
            rank = int(hits[0])
            diagnoses.append((float(row_scores[rank]), rank + 1))
    return diagnoses


def benchmark_authorization(sample_size: int, top_k: int) -> None:
    index, filt, role_sets = build_index()
    doc_of_pos = np.array(index.ids, dtype=object)
    pos_of_doc = {doc_id: pos for pos, doc_id in enumerate(index.ids)}
    all_users = list(role_sets.keys())

    # Candidates are resolved once per distinct role set; every search of a
    # user with that role set reuses them.
    candidates_by_roles = {
        roles: filt.filter_by_roles(list(roles))
        for roles in {tuple(sorted(r)) for r in role_sets.values()}
    }

    role_of_pos = np.full(len(doc_of_pos), None, dtype=object)
    for role in sorted(set().union(*role_sets.values())):
        role_of_pos[filt.filter_by_roles([role])] = role

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

    # (question, scenario) for every applicable scenario, grouped by the
    # user's role set so each group shares one candidate array.
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

    # Warm-up: the first search pays numba's JIT compilation.
    warm_candidates = next(iter(candidates_by_roles.values()))
    index.search_raw(["warm up"], k=top_k, parallel=True, candidates=warm_candidates)

    leaks = 0
    out_of_candidates = 0
    single_recall_hits = 0
    single_recall_checks = 0
    both_found_counts = []
    miss_ranks = []
    miss_scores = []

    bench_start = time.perf_counter()
    for roles, tasks in tqdm(tasks_by_roles.items()):
        candidates = candidates_by_roles[roles]
        if candidates.size == 0:
            ranked = [np.empty(0, dtype=np.int64)] * len(tasks)
        else:
            queries = [q["question"] for q, _ in tasks]
            ranked, _ = _batched_search(index, queries, candidates, min(top_k, candidates.size), BATCH_SIZE)

        misses = []
        for (q, scenario), ranked_pos in zip(tasks, ranked):
            out_of_candidates += int((~np.isin(ranked_pos, candidates)).sum())
            found_ids = set(doc_of_pos[ranked_pos])
            found_a = q["doc_a"] in found_ids
            found_b = q["doc_b"] in found_ids

            if scenario == "none":
                if found_a or found_b:
                    leaks += 1
            elif scenario == "a_only":
                if found_b:
                    leaks += 1
                single_recall_checks += 1
                if found_a:
                    single_recall_hits += 1
                else:
                    misses.append((q["question"], q["pos_a"]))
            elif scenario == "b_only":
                if found_a:
                    leaks += 1
                single_recall_checks += 1
                if found_b:
                    single_recall_hits += 1
                else:
                    misses.append((q["question"], q["pos_b"]))
            elif scenario == "both":
                both_found_counts.append(int(found_a) + int(found_b))

        if misses:
            for score, rank in _diagnose_misses(index, misses, candidates):
                miss_scores.append(score)
                miss_ranks.append(rank)

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
