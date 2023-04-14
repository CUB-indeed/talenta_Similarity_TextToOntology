from typing import Union, List, Tuple

import os
import numpy as np
from .base import BaseSimilarity
from sentence_transformers import SentenceTransformer, util


class BertTransformerSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List], model_name:str="all-MiniLM-L6-v2") -> None:
        """A transformer based neural network for text similarity between two lists on Bert package. 

        Args:
            a (Union[np.array, List]): The first list containing strings.
            b (Union[np.array, List]): The second list containing strings.
            model_name (str, optional): A pretrained model name of text simlarity in Bert Transformer. Defaults to "all-MiniLM-L6-v2".
                Please see the linke [https://www.sbert.net/docs/pretrained_models.html] for more detail.
        """
        super().__init__(a, b)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = SentenceTransformer(model_name)
        self()
    
    def preprocess(self, a:Union[np.array, List], b:Union[np.array, List]) -> Tuple[List, List]:
        doc_function = lambda text: self.model.encode(str(text), convert_to_tensor=True)
        a = list(map(doc_function, a))
        b = list(map(doc_function, b))
        return a, b

    def similarity_func(self, x:str, y:str) -> float:
        cosine_scores = util.pytorch_cos_sim(x, y)
        return cosine_scores.item()