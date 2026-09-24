from simlar import HelixIndex, ReciprocalRankFusion, RelevanceIndex, SimlarEngine, LookupIndex
from bakeoff.contenders.base import Contender

class SimlarHybrid(Contender):
    name = "simlar (HelixIndex)"
    family = "simlar"

    text_weight = 0.1
    vector_weight = .5

    rrf_k = 60

    text_k = 23317           
    vector_k = 200        
    top_k = 100

    def _warm_up(self):
        self.index.search(["warm up the stemmer"], self.matrix[:1], k=1, parallel=True)

    def _build(self, texts, matrix):
        self.index = HelixIndex(
            text_index=RelevanceIndex(),
            vector_index=SimlarEngine(),
            fusion=ReciprocalRankFusion(k=self.rrf_k,
                                        weights=[self.text_weight, self.vector_weight]),
            top_k=self.top_k, vector_k=self.vector_k, text_k=self.text_k)
        self.index.add(ids=self.index_ids, texts=texts, vectors=matrix)

    def _rank(self, query_texts, query_matrix, k):
        hit_lists = self.index.search(query_texts, query_matrix, k=k, parallel=True)
        return [[hit.id for hit in hits] for hits in hit_lists]

class SimlarHybridLookup(SimlarHybrid):
    name = "simlar (Helix with lookup)"
    text_weight = .3
    vector_weight = .9

    text_k = 20_000
    def _build(self, texts, matrix):
        self.index = HelixIndex(
                    text_index=LookupIndex(),
                    vector_index=SimlarEngine(),
                    fusion=ReciprocalRankFusion(k=self.rrf_k,
                                                weights=[self.text_weight, self.vector_weight]),
                    top_k=self.top_k, vector_k=self.vector_k, text_k=self.text_k)
        self.index.add(ids=self.index_ids, texts=texts, vectors=matrix)
