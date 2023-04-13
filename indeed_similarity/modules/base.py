from typing import Union, List, Any

import numpy as np
import pandas as pd
import seaborn as sns


class BaseSimilarity:    
    """
    Find the similarity between 2 lists of string

    Args:
        - a: the first list of string
        - b: the second list of string
    """
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        self.checkInputType(a)
        self.checkInputType(b)
        self.a = a
        self.b = b

    @staticmethod
    def checkInputType(value):
        assert isinstance(value, (list, np.array)), "The input must be either a list or numpy array"
        assert len(value) > 0 , "The list or numpy array is empthy"

    def preprocess(self, a, b):
        return a, b

    def similarity_matrix(self, similarity_func:Any) -> pd.DataFrame:
        a, b = self.preprocess(self.a, self.b)
        arr_sim = np.zeros((len(a), len(b)))  # Create empty matrix to fill

        for i, text1 in enumerate(a):
            for j, text2 in enumerate(b):
                arr_sim[i, j] = similarity_func(text1, text2)
        self.df_sim = pd.DataFrame(arr_sim, columns=self.b, index=self.a)
        return self.df_sim
    
    def plot(self, threshold:int=1):
        assert self.df_sim is not None, "The value of similarity matrix is None"
        df_plot = self.df_sim[self.df_sim<=threshold].fillna(0)
        return sns.heatmap(df_plot)