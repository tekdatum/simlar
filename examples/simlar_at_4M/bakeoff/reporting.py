from pathlib import Path
import pandas as pd
from bakeoff.scoring import EngineResult


class Report:
    """Builds the two tables the benchmark publishes and writes them to disk."""

    def __init__(self, results: list[EngineResult], k_eval: int, k_values: list[int]) -> None:
        if k_eval not in k_values:
            raise ValueError(f"k_eval={k_eval} is not among the scored k values {k_values}")
        self.results = results
        self.k_eval = k_eval
        self.k_values = k_values

    def comparison(self) -> pd.DataFrame:
        """One row per engine: what it found and what it cost, at the headline cutoff."""
        rows = [{
            "index": result.name,
            "family": result.family,
            "Recall accuracy": round(result.recall_at(self.k_eval), 4),
            "ms per query": round(result.ms_per_query, 3),
            "ms per build": round(result.build_ms, 1),
            "ms warm-up": round(result.warm_up_ms, 1),
            "MB in memory": round(result.memory_mb, 1),
        } for result in self.results]
        return (pd.DataFrame(rows).set_index("index")
                .sort_values("Recall accuracy", ascending=False))

    def recall_curve(self) -> pd.DataFrame:
        """Recall accuracy as the cutoff widens -- how much a rerank step could recover."""
        curves = {result.name: result.recall for result in self.results}
        table = pd.DataFrame(curves).T
        table.columns = [f"R_cap@{k}" for k in self.k_values]
        table.index.name = "index"
        return table.round(4).sort_values(f"R_cap@{self.k_eval}", ascending=False)

    def print_all(self, num_queries: int) -> None:
        index_size = self.results[0].index_size if self.results else 0
        print(f"\nsimlar vs everyone -- {num_queries:,} labelled queries against a "
              f"{index_size:,}-passage index, scored at k={self.k_eval}")
        print(self.comparison().to_string())

        print("\nRecall accuracy as the cutoff k widens:")
        print(self.recall_curve().to_string())

    def save(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, frame in (("comparison.csv", self.comparison()),
                            ("recall_curve.csv", self.recall_curve())):
            path = output_dir / name
            frame.to_csv(path)
            written.append(path)
        return written
