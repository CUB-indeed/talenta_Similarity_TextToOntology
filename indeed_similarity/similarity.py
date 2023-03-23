from typing import Any, List, Tuple

import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from Levenshtein import ratio as LevenshteinRatio
from sentence_transformers import SentenceTransformer, util

# Ignore warnings of spacy
import warnings
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)

class Pipeline:
    def __init__(self, similarity_functions:List) -> None:
        self.similarity_functions = similarity_functions

    def __len__(self):
        return len(self.similarity_functions)

    def __call__(self, df1:pd.DataFrame, df2:pd.DataFrame, param:Tuple):
        results = {}
        for func in self.similarity_functions:
            results[func.__name__] = func(df1, df2, param)()
        return results
        

class BaseSimilarity:
    def __init__(self, df1:pd.DataFrame, df2:pd.DataFrame) -> None:
        self.entity_list, self.entity_list2 = df1["name"].values, df2["name"].values
        self.label_list, self.label_list2 = df1["label"].values, df2["label"].values
        self.l1, self.l2 = df1["path"].values, df2["path"].values

    # TODO: param from the old function do nothing. Need to make a filter to filter out the similarity score.
    def similarity_matrix(self, similarity_func:Any, param:Tuple) -> pd.DataFrame:
        #Create empty matrix to fill
        M_sim = np.zeros((self.l1.shape[0], self.l2.shape[0]))

        #Iterate and fill 
        for i in range(self.l1.shape[0]):
            u = self.label_list[i]
            for j in range(self.l2.shape[0]):
                v = self.label_list2[j]
                #similarity -> structural similarity 
                M_sim[i, j] = similarity_func(u, v)
        self.df_sim = pd.DataFrame(M_sim, columns=self.entity_list2, index=self.entity_list)
        return self.df_sim
    
    def plot(self, threshold):
        assert self.df_sim is not None, "The value of similarity matrix is None"
        plot_df = self.df_sim.copy()
        plot_df[plot_df < threshold] = 0.0
        rows, cols = self.df_sim.shape[0:1]

        plt.figure(figsize=((0.4*rows),(0.4*cols)))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        return sns.heatmap(self.df_sim, fmt="g", cmap=cmap,linewidths=0.5, linecolor='black')


class LevenshteinSimilarity(BaseSimilarity):
    def __init__(self, df1:pd.DataFrame, df2:pd.DataFrame, param:Tuple) -> None:
        super().__init__(df1, df2)
        self.param = param

    def __call__(self, *args: Any, **kwds: Any) -> pd.DataFrame:
        return self.similarity_matrix(LevenshteinRatio, self.param)
    

class JaccardSimilarity(BaseSimilarity):
    def __init__(self, df1:pd.DataFrame, df2:pd.DataFrame, param:Tuple) -> None:
        super().__init__(df1, df2)
        self.param = param

    def __call__(self, *args: Any, **kwds: Any) -> pd.DataFrame:
        return self.similarity_matrix(self.jaccard_similarity, self.param)

    def jaccard_similarity(self, x, y):
        """ returns the jaccard similarity between two lists """
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)


class SequenceSimilarity(BaseSimilarity):
    def __init__(self, df1:pd.DataFrame, df2:pd.DataFrame, param:Tuple) -> None:
        super().__init__(df1, df2)
        self.param = param

    def __call__(self, *args: Any, **kwds: Any) -> pd.DataFrame:
        return self.similarity_matrix(self.sequence_similarity, self.param)

    def sequence_similarity(self, x, y):
        return SequenceMatcher(None, x, y).ratio()


class BertTransformerSimilarity(BaseSimilarity):
    def __init__(self, df1:pd.DataFrame, df2:pd.DataFrame, param:Tuple, model_name:str="all-MiniLM-L6-v2") -> None:
        super().__init__(df1, df2)
        self.param = param
        self.model = SentenceTransformer(model_name)

    def __call__(self, *args: Any, **kwds: Any) -> pd.DataFrame:
        return self.similarity_matrix(self.transformers_similarity, self.param)
    
    def transformers_similarity(self, x, y):
        embedding1 = self.model.encode(x, convert_to_tensor=True)
        embedding2 = self.model.encode(y, convert_to_tensor=True)
        # compute similarity scores of two embeddings
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_scores.item()


class SpacyTransformerSimilarity(BaseSimilarity):
    def __init__(self, df1:pd.DataFrame, df2:pd.DataFrame, param:Tuple, model_name:str="en_core_web_md") -> None:
        super().__init__(df1, df2)
        self.param = param
        self.model = spacy.load(model_name)

        self.doc_function = np.vectorize(self.model)
        self.label_list = self.doc_function(df1["label"].values)
        self.label_list2 = self.doc_function(df2["label"].values)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.similarity_matrix(self.transformers_similarity, self.param)
 
    def transformers_similarity(self, x, y):
        return x.similarity(y)
