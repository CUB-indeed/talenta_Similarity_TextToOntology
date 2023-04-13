from typing import Union, List

import os
import numpy as np
from .base import BaseSimilarity
from sentence_transformers import SentenceTransformer, util


class BertTransformerSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List], model_name:str="all-MiniLM-L6-v2") -> None:
        super().__init__(a, b)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
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