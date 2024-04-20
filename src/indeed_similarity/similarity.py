from typing import List, Union, Dict

import pickle
import inspect
import hashlib
import warnings
import pandas as pd
from tqdm import tqdm

from .utils.general import FindFiles, CreateDir
from .modules.base import SimilarityMatrix
from .modules import (
    LevenshteinSimilarity,
    JaccardSimilarity,
    SequenceSimilarity,
    BertTransformerSimilarity,
    SpacyTransformerSimilarity
)

DEFAULT_SIMPIPELINE = [
    LevenshteinSimilarity,
    JaccardSimilarity,
    SequenceSimilarity,
    BertTransformerSimilarity,
    SpacyTransformerSimilarity
]


class SimilarityPipeline:
    def __init__(
        self,
        similarity_functions:List = None,
        preprocessing_functions:List = None,
        postprocessing_functions:List = None,
        ) -> None:
        """A pipeline for calculating multiple text similarities between two lists containing texts.

        Args:
            similarity_functions (List, optional): A list of similarity classes based from BaseSimilarity class. Defaults to None.
            preprocessing_functions (List, optional): A list of preprocessing functions. Defaults to None.
            postprocessing_functions (List, optional): A list of postprocessing functions. Defaults to None.
        """
        self.similarity_functions = similarity_functions if similarity_functions is not None else DEFAULT_SIMPIPELINE
        self.preprocessing_functions = preprocessing_functions
        self.postprocessing_functions = postprocessing_functions

    def __repr__(self) -> str:
        return (
            f"SimilarityPipeline(similarity_functions={self.similarity_functions}, "
            f"preprocessing_functions={self.preprocessing_functions}, "
            f"postprocessing_functions={self.postprocessing_functions})"
        )

    def __getitem__(self, key):
        return self.sim_results[key]

    def __len__(self) -> int:
        return len(self.similarity_functions)

    def __call__(
        self, 
        a:List[str],
        b:List[str],
        cache:bool=False, 
        cache_dir:str="./cache"
        ) -> Dict[str, pd.DataFrame]:
        """_summary_

        Args:
            a (List[str]): The first list containing texts.
            b (List[str]): The second list containing texts.

        Returns:
            Dict[pd.DataFrame]: A dictionary in which keys are the name of similarity classes and values are the result.
        """
        # Cache name
        similarity_functions = [str(func) for func in self.similarity_functions]
        preprocessing_functions = [inspect.getsource(func) for func in self.preprocessing_functions]
        postprocessing_functions = [inspect.getsource(func) for func in self.postprocessing_functions]
        cache_name = list(a) + list(b) + similarity_functions + preprocessing_functions + postprocessing_functions
        cache_name = hashlib.sha256(" ".join(cache_name).encode()).hexdigest() + ".pkl"
        
        # cache path
        cache_dir = CreateDir(cache_dir)
        cache_path = cache_dir / cache_name
        
        if cache and len(FindFiles(cache_dir, formats=[cache_name])) != 0:
            # Load cache
            self.sim_results = pickle.load(open(cache_path, "rb"))
        else:
            self.pre_a, self.pre_b = self.preprocess(a, b)
            self.sim_results = self.process(self.pre_a, self.pre_b)
            self.post_a, self.post_b = self.postprocess()
            self.sim_results = self.__avarage(self.sim_results)  # Add average to the sim_results dict
            if cache:
                pickle.dump(self.sim_results, open(cache_path, "wb"))
        return self.sim_results
    
    
    def preprocess(self, a:List[str], b:List[str]):
        """A function to preprocess two lists of strings

        Args:
            a (List[str]): The first list containing texts.
            b (List[str]): The second list containing texts.

        Returns:
            a (List[str]): The first list containing texts after preprocess.
            b (List[str]): The second list containing texts after preprocess.
        """
        if self.preprocessing_functions is not None:
            for preprocessing_function in self.preprocessing_functions:
                a, b = list(map(preprocessing_function, a)), list(map(preprocessing_function, b))
        return a, b
    
    def process(self, a:List[str], b:List[str]) -> Dict[str, pd.DataFrame]:
        #TODO: Use multi-processing instead of normal for loop.
        """A function to run the pipeline

        Args:
            a (List[str]): The first list containing texts.
            b (List[str]): The second list containing texts.

        Returns:
            Dict[pd.DataFrame]: A dictionary in which keys are the name of similarity classes and values are the result.
        """
        sim_results = {}
        pbar = tqdm(self.similarity_functions)
        for func in pbar:
            pbar.set_description(f"Processing {func.__name__}")
            sim_results[func.__name__] = func(a, b)
        return sim_results
    
    def postprocess(self):
        """A function to postprocess two lists of strings

        Returns:
            a (List[str]): The first list containing texts after postprocess.
            b (List[str]): The second list containing texts after postprocess.
        """
        post_a = []
        post_b = []
        if self.postprocessing_functions is not None:
            if self.preprocessing_functions is None: warnings.warn("There are no transform functions. Please make sure that it is user's intention.")
            sim_mat_temp = self.sim_results[self.similarity_functions[0].__name__].df_sim
            # Get all indexes and columns
            indexes = {text: text for text in sim_mat_temp.index}
            columns = {text: text for text in sim_mat_temp.columns}
            for postprocessing_function in self.postprocessing_functions:
                # Inverse transform indexes and columns
                indexes = {key: postprocessing_function(text) for key, text in indexes.items()}
                columns = {key: postprocessing_function(text) for key, text in columns.items()}
            for similarity_function in self.similarity_functions:
                self.sim_results[similarity_function.__name__].df_sim.rename(index=indexes, inplace=True)
                self.sim_results[similarity_function.__name__].df_sim.rename(columns=columns, inplace=True)
            post_a = list(self.sim_results[self.similarity_functions[0].__name__].df_sim.index)
            post_b = list(self.sim_results[self.similarity_functions[0].__name__].df_sim.columns)
        return post_a, post_b

    @staticmethod
    def __avarage(sim_results) -> pd.DataFrame:
        """An average of similarity matrix of similarity classes

        Returns:
            pd.DataFrame: A dataframe for average of every similarity classes in a form of similarity matrix.
        """
        matrix_names = list(sim_results.keys()).copy()
        matrix_num = len(matrix_names)
        matrix_tt = sim_results[matrix_names.pop(0)].df_sim.copy()
        for matrix_name in matrix_names:
            matrix_tt += sim_results[matrix_name].df_sim
        sim_results["average"] = SimilarityMatrix(matrix_tt/matrix_num)
        return sim_results
