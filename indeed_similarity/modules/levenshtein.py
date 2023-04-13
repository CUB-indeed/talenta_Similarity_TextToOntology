from typing import Union, List

import numpy as np
from .base import BaseSimilarity
from Levenshtein import ratio as LevenshteinRatio


class LevenshteinSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        super().__init__(a, b)
        self.similarity_matrix(LevenshteinRatio)