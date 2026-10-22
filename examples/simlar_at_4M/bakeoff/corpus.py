import gc
from dataclasses import dataclass

from datasets import load_dataset

DATASET_NAME = "microsoft/ms_marco"
DATASET_VERSION = "v2.1"


@dataclass
class Corpus:
    questions: list[str]        # passage_text for a document, the query text for a query
    ids: list[str]

    def __post_init__(self) -> None:
        self._position_by_id = {doc_id: row for row, doc_id in enumerate(self.ids)}
        self._question_by_id = dict(zip(self.ids, self.questions))

    def position_of(self, doc_id: str) -> int:
        return self._position_by_id[doc_id]

    def question_of(self, doc_id: str) -> str:
        return self._question_by_id[doc_id]

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Dataset:
    """A corpus plus the labels that make it scorable."""

    corpus: Corpus
    passage_ids: list[str]      # the documents every engine indexes
    query_ids: list[str]        # the test set
    expected_ids: list[str]     # per query, the passage flagged is_selected

    def texts_for(self, ids: list[str]) -> list[str]:
        return [self.corpus.question_of(doc_id) for doc_id in ids]

    def positions_for(self, ids: list[str]) -> list[int]:
        return [self.corpus.position_of(doc_id) for doc_id in ids]

    def describe(self) -> str:
        return (f"\n{len(self.passage_ids):,} passages to index, "
                f"{len(self.query_ids):,} labelled queries")


class MsMarcoLoader:
    def __init__(self, sample_size: int | None, split: str = "train", queries_cap: int | None = None) -> None:
        self.sample_size = sample_size
        self.split = split
        self.queries_cap = queries_cap

    def load(self) -> Dataset:
        """The corpus, holding both caps: `sample_size` passages and `queries_cap` queries.

        MS MARCO gives about ten passages per labelled row, so a passage budget is spent roughly
        ten times faster than a query budget. Filling it first-come leaves the queries short --
        100k passages yields only ~10k of the 25k queries asked for -- and worse, the row that
        overflows the budget is a query whose own answer did not make it in, which is a query no
        engine can ever get right. So the answers are what the budget is spent on first: a slot is
        held back for every query still to come, and the passages of the rows already read fill
        whatever is left as distractors. Both caps are then met exactly, and -- the reason it
        matters for a sweep over index sizes -- every size is scored against the same queries
        against the same answers, so a difference in recall is the index size and nothing else.
        """
        if self.sample_size is not None and self.queries_cap is not None \
                and self.queries_cap > self.sample_size:
            # Refused before the scan rather than discovered 4 GB in: every query's answer is one
            # of the indexed passages, so more queries than passages cannot be scored.
            raise ValueError(f"queries_cap={self.queries_cap} exceeds sample_size="
                             f"{self.sample_size}; a query's answer has to be in the index")

        rows = load_dataset(DATASET_NAME, DATASET_VERSION, split=self.split)
        gc.disable()

        text_hash_to_pid: dict[int, str] = {}
        unique_passages: list[str] = []
        unique_passage_ids: list[str] = []

        queries: list[str] = []
        expected_ids: list[str] = []
        unanswerable = 0        # queries dropped because their answer would not fit in the index

        def indexed_id(text: str, reserve: int = 0) -> str | None:
            """The id of `text` in the index, indexing it if it is new and there is room for it.

            None when the budget is full: the caller decides whether that costs a distractor,
            which is nothing, or a query, which is the query. `reserve` is the room to leave
            unspent -- the answers still to come -- so a distractor cannot take their place.
            """
            key = hash(text)
            if key in text_hash_to_pid:
                return text_hash_to_pid[key]        # a duplicate costs no budget at all
            if self.sample_size is not None and len(unique_passages) + reserve >= self.sample_size:
                return None
            pid = f"p_{len(unique_passages)}"
            text_hash_to_pid[key] = pid
            unique_passages.append(text)
            unique_passage_ids.append(pid)
            return pid

        try:
            for row in rows:
                candidates = row["passages"]
                selected = candidates["is_selected"]

                selected_indices = [i for i, flag in enumerate(selected) if flag]
                if not selected_indices:
                    continue

                passage_texts = candidates["passage_text"]
                # Several passages can be flagged relevant. The first is the one scored against,
                # because recall only ever credits a single expected id.
                answer_position = selected_indices[0]
                wanted_query = self.queries_cap is None or len(queries) < self.queries_cap

                # In row order, so that a corpus whose budgets never bind is the same corpus it
                # has always been -- same passages, same ids, and so the same cached embeddings.
                for position, text in enumerate(passage_texts):
                    text_str = str(text)
                    if position == answer_position and wanted_query:
                        answer_id = indexed_id(text_str)    # reserve nothing: this is the answer
                        if answer_id is None:
                            unanswerable = unanswerable + 1
                            continue
                        queries.append(str(row["query"]))
                        expected_ids.append(answer_id)
                    else:
                        # Room kept for every answer still to come, counted fresh each time so
                        # that this row's own answer is covered until it has actually been added.
                        remaining = (0 if self.queries_cap is None
                                     else max(0, self.queries_cap - len(queries)))
                        indexed_id(text_str, reserve=remaining)

                if (self.sample_size is not None and len(unique_passages) >= self.sample_size
                        and self.queries_cap is not None and len(queries) >= self.queries_cap):
                    break

        finally:
            gc.enable()

        # What was asked for and what the split could give are not always the same. Said out loud
        # here, because the alternative is a recall column quietly measured on a different test set.
        if self.queries_cap is not None and len(queries) < self.queries_cap:
            print(f"  Warning: asked for {self.queries_cap:,} queries, the {self.split} split "
                  f"gave {len(queries):,}", flush=True)
        if self.sample_size is not None and len(unique_passages) < self.sample_size:
            print(f"  Warning: asked for {self.sample_size:,} passages, the {self.split} split "
                  f"gave {len(unique_passages):,}", flush=True)
        if unanswerable:
            print(f"  Warning: {unanswerable:,} queries skipped, their answer did not fit in a "
                  f"{self.sample_size:,}-passage index", flush=True)

        query_ids = [f"q_{index}" for index in range(len(queries))]

        corpus = Corpus(
            questions=unique_passages + queries, 
            ids=unique_passage_ids + query_ids
        )
        
        return Dataset(
            corpus=corpus, 
            passage_ids=unique_passage_ids, 
            query_ids=query_ids, 
            expected_ids=expected_ids
        )
