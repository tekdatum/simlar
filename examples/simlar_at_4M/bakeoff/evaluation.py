"""Building, timing and scoring engines against one corpus.

This is the whole of a specific evaluation: hand it a dataset, the vectors and the cutoffs, give
it a list of contenders, and get an EngineResult each.
"""
import traceback
from collections.abc import Sequence

import numpy as np

from bakeoff.corpus import Dataset
from bakeoff.contenders.base import Contender
from bakeoff.scoring import Cutoffs, EngineResult


class BakeOff:
    """One corpus, embedded once, against as many engines as you like.

    Everyone indexes the same passages with the same embeddings and is scored against the same
    labelled queries, which is the only reason the numbers can be put in one table.
    """

    def __init__(self, dataset: Dataset, vectors: np.ndarray, k_max: int,
                 k_values: list[int]) -> None:
        self.dataset = dataset
        # Refused here rather than at the report: a cutoff deeper than the ranked lists cannot be
        # scored, and finding that out afterwards is an hour of building thrown away.
        self.cutoffs = Cutoffs(k_max, k_values)

        self.index_texts = dataset.texts_for(dataset.passage_ids)
        self.index_matrix = np.ascontiguousarray(
            vectors[dataset.positions_for(dataset.passage_ids)])
        self.query_texts = dataset.texts_for(dataset.query_ids)
        self.query_matrix = np.ascontiguousarray(
            vectors[dataset.positions_for(dataset.query_ids)])

    @property
    def k_max(self) -> int:
        return self.cutoffs.k_max

    @property
    def k_values(self) -> list[int]:
        return list(self.cutoffs.k_values)

    @property
    def index_size(self) -> int:
        """Passages in the index -- what a row was measured against, not with."""
        return len(self.dataset.passage_ids)

    @property
    def queries(self) -> int:
        """Labelled queries scoring it. Two corpora scored by different query sets are two tests."""
        return len(self.dataset.query_ids)

    def run(self, contender_types: Sequence[type[Contender]]) -> list[EngineResult]:
        return [self.run_one(contender_type) for contender_type in contender_types]

    def run_one(self, contender_type: type[Contender]) -> EngineResult:
        """Build, warm and score one engine, surviving whatever it does on the way down.

        An engine that dies takes only its own row with it: a database that will not start, or
        an index that runs the machine out of memory, should not throw away the engines already
        measured or the ones still queued behind it.

        The row starts at zero and each field is filled once it has actually been earned, so
        whatever the failure interrupted keeps its zero -- an engine that built but could not
        answer still reports its build cost and footprint, against zero recall.
        """
        print(f"--- {contender_type.name} ---", flush=True)
        result = EngineResult(
            name=contender_type.name, family=contender_type.family,
            index_size=self.index_size,
            ms_per_query=0.0, build_ms=0.0, warm_up_ms=0.0, memory_mb=0.0,
            recall=self.cutoffs.nothing_found(),
        )
        try:
            engine = contender_type(self.dataset.passage_ids, self.index_texts,
                                    self.index_matrix)
            result.build_ms = engine.build()
            result.memory_mb = engine.memory_mb
            print(f"  built  {engine.name:32s} [{engine.family:10s}] "
                  f"{engine.size:,} passages in {result.build_ms:8.1f} ms "
                  f"({result.memory_mb:6.1f} MB)", flush=True)
            result.warm_up_ms = engine.warm_up()    # one-time setup, outside every stopwatch

            ranked = engine.search(self.query_texts, self.query_matrix, self.k_max)
            # Scored first, then both assigned: a latency for answers too malformed to score is
            # not a measurement of anything, and would otherwise plot as a real point at 0 recall.
            recall = self.cutoffs.recall_of(ranked, self.dataset.expected_ids)
            result.ms_per_query, result.recall = engine.ms_per_query, recall
            print(f"  ranked {engine.name:32s} {result.ms_per_query:8.3f} ms/query", flush=True)
            print(f"  recall {engine.name:32s} "
                  f"{', '.join(f'@{k}={recall[k]:.3f}' for k in self.k_values)}", flush=True)
        except Exception as error:
            # Not BaseException: a Ctrl-C should still stop the run, not skip to the next engine.
            print(f"  FAILED {result.name:32s} {type(error).__name__}: {error}", flush=True)
            # traceback.print_exc()       # to stderr, so the run log above stays scannable
        return result
