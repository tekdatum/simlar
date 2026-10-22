import faiss

from bakeoff.contenders.base import Contender
from bakeoff.environment import available_cores

faiss.omp_set_num_threads(available_cores())

class FaissContender(Contender):
    name = "faiss (IndexFlatIP)"
    family = "dense"

    def _warm_up(self):
        self.index.search(self.matrix[:1], 1)

    def _build(self, texts, matrix):
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def _rank(self, query_texts, query_matrix, k):
        _scores, positions = self.index.search(query_matrix, k)
        return [self.ids_at(row) for row in positions]
