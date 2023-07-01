from typing import Union, List, Any, Tuple, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

SIM_STACK_COLUMNS = ["onto1", "onto2", "confidence"]

class SimilarityMatrix:
    def __init__(self, df_sim) -> None:
        self.df_sim = df_sim

    def plot(self, threshold:int=0, plotly=False, **kwargs) -> sns:
        """A function to plot similarity matrix in heatmap.

        Args:
            threshold (int, optional): A threshold to filter out confidence score. Defaults to 1.

        Returns:
            sns: A heatmap of similarity matrix
        """
        vmin, vmax = 0, 1
        assert self.df_sim is not None, "The value of similarity matrix is None"
        df_plot = self.df_sim[self.df_sim>=threshold]
        df_plot = df_plot.dropna(axis=0, how="all")
        df_plot = df_plot.dropna(axis=1, how="all")
        if plotly:
            data = go.Heatmap(x=df_plot.columns, y=df_plot.index, z=df_plot.to_numpy(), zmin=vmin, zmax=vmax, **kwargs)
            return go.Figure(data=data)
        else:
            return sns.heatmap(df_plot, vmin=vmin, vmax=vmax, **kwargs)
    
    @property
    def df_sim_stack(self) -> pd.DataFrame:
        """Convert from similarity matrix to stack dataframe.

        Args:
            mat (pd.DataFrame): A similarity matrix dataframe

        Returns:
            pd.DataFrame: A stack dataframe
        """
        df = self.df_sim.stack().reset_index()
        df.columns = SIM_STACK_COLUMNS
        return df


class BaseSimilarity(SimilarityMatrix):
    def __init__(self, a:List[str], b:List[str]) -> None:
        """A class for calculating text similarities between two lists containing strings.

        Args:
            a (List[str]): The first list containing strings.
            b (List[str]): The second list containing strings.
        """
        self.___checkInputType(a)
        self.___checkInputType(b)
        self.a = a
        self.b = b

    @staticmethod
    def ___checkInputType(value:List[str]) -> None:
        """Check the input type.

        Args:
            value (List[str]): A list containing strings.
        """
        assert isinstance(value, (list, np.array)), "The input must be either a list or arraylike"
        assert len(value) > 0 , "The list or numpy array is empthy"

    def preprocess(self, a:List[str], b:List[str]) -> Tuple[List, List]:
        """A preprocess step to deal with text before calculating similarity matrix.

        Args:
            a (List[str]): The first list containing strings.
            b (List[str]): The second list containing strings.

        Returns:
            a (List[str]): The first list containing strings.
            b (List[str]): The second list containing strings.
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
        df_sim = pd.DataFrame(arr_sim, columns=self.b, index=self.a)
        super().__init__(df_sim)
        return df_sim

    def __str__(self) -> str:
        return str(self.__class__.__name__)
