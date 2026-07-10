from __future__ import annotations

from simlar.contracts import SearchResult

class ReciprocalRankFusion:
    def __init__(self, k: int = 2, weights: list[float] | None = None) -> None: ...
    def __call__(self, results: list[list[SearchResult]], k: int) -> list[SearchResult]: ...
