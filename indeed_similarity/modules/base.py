from typing import Union, List, Any, Tuple

import numpy as np
import pandas as pd
import seaborn as sns


class BaseSimilarity:
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        """A class for calculating text similarities between two lists containing strings.

        Args:
            a (Union[np.array, List]): The first list containing strings.
            b (Union[np.array, List]): The second list containing strings.
        """
        self.checkInputType(a)
        self.checkInputType(b)
        self.a = a
        self.b = b

    @staticmethod
    def checkInputType(value:Union[np.array, List]) -> None:
        """Check the input type.

        Args:
            value (Union[np.array, List]): A list containing strings.
        """
        assert isinstance(value, (list, np.array)), "The input must be either a list or numpy array"
        assert len(value) > 0 , "The list or numpy array is empthy"

    def preprocess(self, a:Union[np.array, List], b:Union[np.array, List]) -> Tuple[List, List]:
        """A preprocess step to deal with text before calculating similarity matrix.

        Args:
            a (Union[np.array, List]): The first list containing strings.
            b (Union[np.array, List]): The second list containing strings.

        Returns:
            a (Union[np.array, List]): The first list containing strings.
            b (Union[np.array, List]): The second list containing strings.
        """
        return a, b

    def similarity_func(self, x:str, y:str) -> float:
        """A function to calulate text similarity confidence between two strings.

        Args:
            x (str): The first string.
            y (str): The second string.

        Returns:
            float: A text similarity confidence value.
        """
        return 1 if x==y else 0

    def __call__(self) -> pd.DataFrame:
        """A function to calculate a similarity matrix.

        Returns:
            pd.DataFrame: A similarity matrix
        """
        a, b = self.preprocess(self.a, self.b)
        arr_sim = np.zeros((len(a), len(b)))  # Create empty matrix to fill

        for i, text1 in enumerate(a):
            for j, text2 in enumerate(b):
                arr_sim[i, j] = self.similarity_func(text1, text2)
        self.df_sim = pd.DataFrame(arr_sim, columns=self.b, index=self.a)
        return self.df_sim
    
    def plot(self, threshold:int=0) -> sns:
        """A function to plot similarity matrix in heatmap.

        Args:
            threshold (int, optional): A threshold to filter out confidence score. Defaults to 1.

        Returns:
            sns: A heatmap of similarity matrix
        """
        assert self.df_sim is not None, "The value of similarity matrix is None"
        df_plot = self.df_sim[self.df_sim<=threshold].fillna(0)
        return sns.heatmap(df_plot)