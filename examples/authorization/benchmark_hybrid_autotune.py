"""Seed HotpotQA (multi-hop QA) and measure authorization under partial
context access: each question needs 2 supporting paragraphs, so a user can
have access to 0, 1 (either one), or both of them.

Hybrid variant of hotpotqa.py / hotpotqa_bm25.py: a HelixIndex (BM25 ->
vectors restricted to the BM25 hits -> RRF), with text_k / vector_k / top_k
taken from HybridTunerLite instead of being hand-picked.

  - document_sections.content + embedding -> HelixIndex (RelevanceIndex + SimlarEngine)
  - document_access / roles               -> SQLFilter (grant / filter_by_roles)
  - users / user_roles                    -> in-memory dict (user -> role names)
  - search_document_sections(user, ...)   -> filter_by_roles(user's roles) as
                                             `candidates=` of the HelixIndex search

The tuner runs once, before the index is built: its depths only affect recall
and speed, never which documents a user may see (that is `candidates=`). The
corpus it is asked about is the mean authorized set size, since that is what
each search actually covers. The fusion weights are not predicted by the
tuner, and its model was measured at [0.1, 0.5]: with other weights its depths
and expected recall are only a reference.

Searches go through HelixIndex's own cascade (`_retrieve_raw`, with
`candidates=`) plus its fusion's `fuse_batch` -- the same ranking as
HelixIndex.search, row for row, but as positions instead of SearchResult
objects, and with each arm's hits kept so a miss can be traced to the arm
that dropped it.

No worker threads: RelevanceIndex tokenizes queries with a PyStemmer
instance, which must not be called concurrently. Searches that share a
candidate set go through batched calls with parallel=True instead.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from simlar import HelixIndex, ReciprocalRankFusion, RelevanceIndex, SimlarEngine
from simlar_engine import SQLFilter
from tqdm import tqdm

TUNER_DIR = Path(__file__).resolve().parent / "playground" / "simlar_vs_everyone" / "HybridTunerLite"
sys.path.insert(0, str(TUNER_DIR))
from hybrid_tuner_lite import HybridTunerLite  # noqa: E402

NUM_QUESTIONS = 90447  # full train split
NUM_ROLES = 10
NUM_USERS = 100
BATCH_SIZE = 128  # queries per search batch; each holds text_k ids + scores per query

# The model and the fusion it was measured with: relevance.npz holds for BM25
# (robertson, k1=1.5, b=0.75) fused with RRF k=60 at weights [0.1, 0.5].
TUNER_MODEL = TUNER_DIR / "models" / "relevance.npz"
SPEED_PREFERENCE = "fastest"
TOP_K = 100  # results per search; the cutoff the tuner is asked about
RRF_K = 60
TEXT_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

model = SentenceTransformer("all-MiniLM-L6-v2")


def _doc_id(question_idx: int, paragraph_idx: int) -> str:
    """Deterministic paragraph id, so gold paragraphs resolve without a title lookup."""
    return f"{question_idx}:{paragraph_idx}"


def build_index() -> tuple[HelixIndex, ReciprocalRankFusion, SQLFilter, dict, list[str], float | None]:
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
    role_sets = {user_id: set(roles) for user_id, roles in users.items()}

    print("Building SQLFilter (document_access)...")
    filt = SQLFilter()
    filt.add(doc_ids)
    docs_by_role: dict[str, list[str]] = {}
    for doc_id, role in zip(doc_ids, doc_roles):
        docs_by_role.setdefault(role, []).append(doc_id)
    for role, ids in docs_by_role.items():
        filt.grant(ids, [role])

    # One tuner call for the whole run, at the size of the corpus a search covers.
    authorized_sizes = [
        filt.filter_by_roles(list(roles)).size
        for roles in {tuple(sorted(r)) for r in role_sets.values()}
    ]
    corpus_size = round(float(np.mean(authorized_sizes)))
    settings = HybridTunerLite(TUNER_MODEL, use_cache=False).predict(
        corpus_size=corpus_size, cutoff=TOP_K, speed_preference=SPEED_PREFERENCE
    )
    print(
        f"text_k={settings.text_k:,} vector_k={settings.vector_k} top_k={settings.top_k} "
        f"expected_recall={settings.expected_recall} TOP_K={TOP_K} "
        f"SPEED_PREFERENCE={SPEED_PREFERENCE} RRF_K={RRF_K} "
        f"TEXT_WEIGHT={TEXT_WEIGHT} VECTOR_WEIGHT={VECTOR_WEIGHT}"
    )

    print("Generating embeddings...")
    embeddings = model.encode(doc_contents, batch_size=64, show_progress_bar=True)

    # Same ids, same order -> index positions and filter positions line up.
    print("Building HelixIndex (BM25 + SimlarEngine)...")
    fusion = ReciprocalRankFusion(k=RRF_K, weights=[TEXT_WEIGHT, VECTOR_WEIGHT])
    index = HelixIndex(
        text_index=RelevanceIndex(),
        vector_index=SimlarEngine(),
        fusion=fusion,
        text_k=settings.text_k,
        vector_k=settings.vector_k,
        top_k=settings.top_k,
    )
    index.add(doc_ids, doc_contents, np.asarray(embeddings, dtype=np.float32))
    del doc_contents, embeddings

    print(f"Index build complete: {len(doc_ids)} paragraphs.")
    return index, fusion, filt, role_sets, doc_ids, settings.expected_recall


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
    index: HelixIndex,
    fusion: ReciprocalRankFusion,
    queries: list[str],
    embeddings: np.ndarray,
    candidates: np.ndarray,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Hybrid search of many queries restricted to one shared, flat candidate set.

    Returns (fused, text_ids, vector_ids) per query, as positions with the -1
    padding dropped, so a miss can be traced to the arm that dropped it.
    """
    fused_all, text_all, vector_all = [], [], []
    for start in range(0, len(queries), batch_size):
        text_ids, _, vector_ids, _ = index._core._retrieve_raw(
            queries[start : start + batch_size],
            embeddings[start : start + batch_size],
            TOP_K,
            True,
            candidates=candidates,
        )
        fused_ids, _ = fusion.fuse_batch(text_ids, vector_ids, TOP_K, parallel=True)
        fused_all.extend(row[row >= 0] for row in fused_ids)
        text_all.extend(text_ids)
        vector_all.extend(vector_ids)
    return fused_all, text_all, vector_all


def benchmark_authorization(sample_size: int) -> None:
    index, fusion, filt, role_sets, doc_ids, expected_recall = build_index()
    doc_of_pos = np.array(doc_ids, dtype=object)
    pos_of_doc = {doc_id: pos for pos, doc_id in enumerate(doc_ids)}
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

    embeddings = model.encode([q["question"] for q in questions], batch_size=64)
    for q, embedding in zip(questions, embeddings):
        q["embedding"] = np.asarray(embedding, dtype=np.float32)

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

    # Warm-up: the first search pays numba's JIT compilation. Two queries, so
    # HelixIndex takes its batch path, the one the benchmark measures.
    warm_candidates = next(c for c in candidates_by_roles.values() if c.size > 0)
    warm_vecs = np.stack([questions[0]["embedding"]] * 2)
    _batched_search(index, fusion, ["warm up"] * 2, warm_vecs, warm_candidates, BATCH_SIZE)

    leaks = 0
    out_of_candidates = 0
    single_recall_hits = 0
    single_recall_checks = 0
    both_found_counts = []
    # Where each Recall 1 miss was dropped: not in the BM25 hits, cut by the
    # vector arm, or present in both arms but outranked in the fusion.
    dropped_at = {"text": 0, "vector": 0, "fusion": 0}

    bench_start = time.perf_counter()
    for roles, tasks in tqdm(tasks_by_roles.items()):
        candidates = candidates_by_roles[roles]
        if candidates.size == 0:
            ranked = text_ranked = vector_ranked = [np.empty(0, dtype=np.int64)] * len(tasks)
        else:
            queries = [q["question"] for q, _ in tasks]
            query_vecs = np.stack([q["embedding"] for q, _ in tasks])
            ranked, text_ranked, vector_ranked = _batched_search(
                index, fusion, queries, query_vecs, candidates, BATCH_SIZE
            )

        for (q, scenario), ranked_pos, text_pos, vector_pos in zip(tasks, ranked, text_ranked, vector_ranked):
            out_of_candidates += int((~np.isin(ranked_pos, candidates)).sum())
            found_ids = set(doc_of_pos[ranked_pos])
            found_a = q["doc_a"] in found_ids
            found_b = q["doc_b"] in found_ids

            missed_pos = None
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
                    missed_pos = q["pos_a"]
            elif scenario == "b_only":
                if found_a:
                    leaks += 1
                single_recall_checks += 1
                if found_b:
                    single_recall_hits += 1
                else:
                    missed_pos = q["pos_b"]
            elif scenario == "both":
                both_found_counts.append(int(found_a) + int(found_b))

            if missed_pos is not None:
                if missed_pos not in text_pos:
                    dropped_at["text"] += 1
                elif missed_pos not in vector_pos:
                    dropped_at["vector"] += 1
                else:
                    dropped_at["fusion"] += 1

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
    print(f"Tuner expected recall (vs Recall 1): "
          f"{'n/a' if expected_recall is None else f'{100 * expected_recall:.1f}%'}")
    print(f"Recall 1 misses dropped at: text={dropped_at['text']} "
          f"vector={dropped_at['vector']} fusion={dropped_at['fusion']}")
    print(f"Avg time: {avg_time * 1000:.1f}ms")
    print(f"Total time: {total_elapsed / 60:.1f}min")


if __name__ == "__main__":
    benchmark_authorization(sample_size=10_000)
