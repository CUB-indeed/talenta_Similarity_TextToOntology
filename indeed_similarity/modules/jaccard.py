from typing import Union, List

import numpy as np
from .base import BaseSimilarity


class JaccardSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        super().__init__(a, b)
        self.similarity_matrix(self.jaccard_similarity)

    def jaccard_similarity(self, x, y):
        """ returns the jaccard similarity between two lists """
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)