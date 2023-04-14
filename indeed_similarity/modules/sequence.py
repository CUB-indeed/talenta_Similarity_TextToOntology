from typing import Union, List

import numpy as np
from .base import BaseSimilarity
from difflib import SequenceMatcher


class SequenceSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        """Sequence similarity between two lists.

        Args:
            a (Union[np.array, List]): The first list containing strings.
            b (Union[np.array, List]): The second list containing strings.
        """
        super().__init__(a, b)
        self()

    def similarity_func(self, x:str, y:str) -> float:
        return SequenceMatcher(None, x, y).ratio()