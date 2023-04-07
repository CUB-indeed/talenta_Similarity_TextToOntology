from typing import Any, List, Tuple, Union

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

class SimilarityPipeline:
    """
    A pipeline for multiple similarity functions

    Args:
        - similarity_functions: A list of similarity funtions based on BaseSimilarity Class
    """
    def __init__(self, similarity_functions:List) -> None:
        self.similarity_functions = similarity_functions

    def __len__(self):
        return len(self.similarity_functions)

    def __call__(self, a:Union[np.array, List], b:Union[np.array, List]):
        results = {}
        for func in self.similarity_functions:
            results[func.__name__] = func(a, b)
        return results


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


class LevenshteinSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        super().__init__(a, b)
        self.similarity_matrix(LevenshteinRatio)


class JaccardSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        super().__init__(a, b)
        self.similarity_matrix(self.jaccard_similarity)

    def jaccard_similarity(self, x, y):
        """ returns the jaccard similarity between two lists """
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)


class SequenceSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List]) -> None:
        super().__init__(a, b)
        self.similarity_matrix(self.sequence_similarity)

    def sequence_similarity(self, x, y):
        return SequenceMatcher(None, x, y).ratio()


class BertTransformerSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List], model_name:str="all-MiniLM-L6-v2") -> None:
        super().__init__(a, b)
        self.model = SentenceTransformer(model_name)
        self.similarity_matrix(self.transformers_similarity)
    
    def preprocess(self, a, b):
        doc_function = lambda text: self.model.encode(str(text), convert_to_tensor=True)
        a = list(map(doc_function, a))
        b = list(map(doc_function, b))
        return a, b

    def transformers_similarity(self, x, y):
        cosine_scores = util.pytorch_cos_sim(x, y)
        return cosine_scores.item()


class SpacyTransformerSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List], model_name:str="en_core_web_md") -> None:
        super().__init__(a, b)
        self.model = spacy.load(model_name)
        self.similarity_matrix(self.transformers_similarity)

    def preprocess(self, a, b):
        doc_function = lambda text: self.model(str(text))
        a = list(map(doc_function, a))
        b = list(map(doc_function, b))
        return a, b

    def transformers_similarity(self, x, y):
        return x.similarity(y)
