"""What one measurement is, and how it is scored.

Kept apart from the machinery that produces it because everything downstream reads it and nothing
downstream cares how it was made: the report, the bubble chart and the sweep's CSV all take an
EngineResult and go their own way.
"""
from dataclasses import dataclass, field


@dataclass
class EngineResult:
    """One engine, measured against one corpus: what it found and what it cost."""

    name: str
    family: str
    index_size: int
    ms_per_query: float
    build_ms: float
    warm_up_ms: float           # one-off first-call setup, paid off the clock
    memory_mb: float            # resident memory the built index added to the process
    recall: dict[int, float] = field(default_factory=dict)   # recall at each k

    def recall_at(self, k: int) -> float:
        return self.recall[k]


@dataclass(frozen=True)
class Cutoffs:
    """How deep to rank, and which depths to score -- checked once, before anything is built.

    `k_max` is the one ranking every engine actually performs; every cutoff at or below it is read
    off those same lists, which is why widening a report costs nothing. A cutoff deeper than the
    lists cannot be scored at all, and finding that out at the report is an hour of building
    thrown away, so it is refused here instead.
    """

    k_max: int
    k_values: tuple[int, ...]

    def __post_init__(self) -> None:
        # A list is what every caller has to hand; a tuple is what a frozen value object should
        # hold, so the conversion happens here rather than at four call sites.
        object.__setattr__(self, "k_values", tuple(self.k_values))
        too_deep = [k for k in self.k_values if k > self.k_max]
        if too_deep:
            raise ValueError(f"k values {too_deep} are deeper than k_max={self.k_max}; "
                             f"rank at least that deep or drop them from k_values")

    def nothing_found(self) -> dict[int, float]:
        """A zero at every scored cutoff: what an engine that never answered is credited with."""
        return dict.fromkeys(self.k_values, 0.0)

    def recall_of(self, ranked_lists: list[list[str]], expected_ids: list[str]) -> dict[int, float]:
        """The fraction of queries whose one expected passage landed above each cutoff."""
        found = dict.fromkeys(self.k_values, 0)
        for expected_id, ranked in zip(expected_ids, ranked_lists):
            rank = next((i for i, doc_id in enumerate(ranked) if doc_id == expected_id), None)
            if rank is None:
                continue
            for k in self.k_values:
                found[k] += rank < k
        total = len(expected_ids)
        return {k: found[k] / total for k in self.k_values}
