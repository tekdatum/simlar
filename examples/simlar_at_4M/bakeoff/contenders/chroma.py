import contextlib

import chromadb

from bakeoff.contenders.base import Contender
from bakeoff.environment import available_cores

COLLECTION = "passages"
QUERY_CAP = 1000


def in_slices(total: int, cap: int):
    """Yield (start, stop) windows of at most `cap` rows."""
    for start in range(0, total, cap):
        yield start, min(start + cap, total)


class ChromaContender(Contender):
    name = "chroma"
    family = "vector-db"

    def _warm_up(self):
        self.index.query(query_embeddings=self.matrix[:1].tolist(), n_results=1)

    def _build(self, texts, matrix):
        client = chromadb.Client()
        with contextlib.suppress(Exception):
            client.delete_collection(COLLECTION)        # no-op on the first run
        self.index = client.create_collection(
            COLLECTION,
            configuration={"hnsw": {"space": "cosine", "num_threads": available_cores()}})
        for start, stop in in_slices(self.size, client.get_max_batch_size()):
            self.index.add(ids=[str(position) for position in range(start, stop)],
                           embeddings=matrix[start:stop].tolist())

    def _rank(self, query_texts, query_matrix, k):
        ranked = []
        for start, stop in in_slices(len(query_matrix), QUERY_CAP):
            hits = self.index.query(query_embeddings=query_matrix[start:stop].tolist(),
                                    n_results=k)
            ranked.extend(self.ids_at(row) for row in hits["ids"])
        return ranked
