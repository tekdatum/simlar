import numpy as np
from turbovec import TurboQuantIndex

from bakeoff.contenders.base import Contender

BIT_WIDTH = 4        # bits each vector dimension is squeezed down to


class TurboVecContender(Contender):
    name = "turbovec"
    family = "dense"

    def _warm_up(self):
        self.index.search(self.matrix[:1], k=1)

    def _build(self, texts, matrix):
        self.index = TurboQuantIndex(dim=matrix.shape[1], bit_width=BIT_WIDTH)
        self.index.add(matrix)

    def _rank(self, query_texts, query_matrix, k):
        _scores, positions = self.index.search(query_matrix, k=k)
        return [self.ids_at(row) for row in np.asarray(positions)]
