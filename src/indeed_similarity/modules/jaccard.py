from typing import Union, List

import numpy as np
from .base import BaseSimilarity


class JaccardSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        """Jaccard similarity between two lists.

        Args:
            a (Union[np.array, List]): The first list containing strings.
            b (Union[np.array, List]): The second list containing strings.
        """
        super().__init__(a, b)
        self()

    def similarity_func(self, x:str, y:str) -> float:
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)