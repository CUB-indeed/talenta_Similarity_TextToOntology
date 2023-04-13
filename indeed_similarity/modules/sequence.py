from typing import Union, List

import numpy as np
from .base import BaseSimilarity
from difflib import SequenceMatcher


class SequenceSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        super().__init__(a, b)
        self.similarity_matrix(self.sequence_similarity)

    def sequence_similarity(self, x, y):
        return SequenceMatcher(None, x, y).ratio()