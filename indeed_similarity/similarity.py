from typing import List, Union

import warnings
import numpy as np
from tqdm import tqdm

from .modules.levenshtein import LevenshteinSimilarity
from .modules.jaccard import JaccardSimilarity
from .modules.sequence import SequenceSimilarity
from .modules.bert import BertTransformerSimilarity
from .modules.spacy import SpacyTransformerSimilarity

DEFAULT_SIMPIPELINE = [
        LevenshteinSimilarity,
        JaccardSimilarity,
        SequenceSimilarity,
        BertTransformerSimilarity,
        SpacyTransformerSimilarity
    ]

class SimilarityPipeline:
    """
    A pipeline for multiple similarity functions

    Args:
        - similarity_functions: A list of similarity funtions based on BaseSimilarity Class
    """
    def __init__(
        self,
        a:Union[np.array, List],
        b:Union[np.array, List],
        similarity_functions:List = None, 
        preprocessing_functions:List = None,
        postprocessing_functions:List = None,
        ) -> None:
        self.similarity_functions = similarity_functions if similarity_functions is not None else DEFAULT_SIMPIPELINE
        self.a, self.b = a, b
        if preprocessing_functions is not None:
            for preprocessing_function in preprocessing_functions:
                a, b = list(map(preprocessing_function, a)), list(map(preprocessing_function, b))
        self.pre_a, self.pre_b = a, b
        self.sim_results = self(a, b)
        if postprocessing_functions is not None:
            if preprocessing_functions is None: warnings.warn("There are no transform functions. Please make sure that it is user's intention.")
            sim_mat_temp = self.sim_results[self.similarity_functions[0].__name__].df_sim
            # Get all indexes and columns
            indexes = {text: text for text in sim_mat_temp.index}
            columns = {text: text for text in sim_mat_temp.columns}
            for postprocessing_function in postprocessing_functions:
                # Inverse transform indexes and columns
                indexes = {key: postprocessing_function(text) for key, text in indexes.items()}
                columns = {key: postprocessing_function(text) for key, text in columns.items()}
            for similarity_function in similarity_functions:
                self.sim_results[similarity_function.__name__].df_sim.rename(index=indexes, inplace=True)
                self.sim_results[similarity_function.__name__].df_sim.rename(columns=columns, inplace=True)
        self.post_a = list(self.sim_results[self.similarity_functions[0].__name__].df_sim.index)
        self.post_b = list(self.sim_results[self.similarity_functions[0].__name__].df_sim.columns)

    def __len__(self):
        return len(self.similarity_functions)

    def __call__(self, a:Union[np.array, List], b:Union[np.array, List]):
        sim_results = {}
        pbar = tqdm(self.similarity_functions)
        for func in pbar:
            sim_results[func.__name__] = func(a, b)
            pbar.set_description(f"Processing {func.__name__}")
        return sim_results

    @staticmethod
    def mat_sim2stack(mat):
        df = mat.stack().reset_index()
        df.columns = ["onto1", "onto2", "confidence"]
        return df
    
    @property
    def sim_mat(self):
        return dict((name, sim_result.df_sim) for name, sim_result in self.sim_results.items())

    @property
    def sim_mat_stack(self):
        return dict((name, self.mat_sim2stack(sim_result.df_sim)) for name, sim_result in self.sim_results.items())

    @property
    def sim_mat_avg(self):
        matrix_names = list(self.sim_mat.keys()).copy()
        matrix_num = len(matrix_names)
        matrix_tt = self.sim_mat[matrix_names.pop(0)].copy()
        for matrix_name in matrix_names:
            matrix_tt += self.sim_mat[matrix_name]
        return matrix_tt/matrix_num
    
    @property
    def sim_mat_avg_stack(self):
        return self.mat_sim2stack(self.sim_mat_avg)
