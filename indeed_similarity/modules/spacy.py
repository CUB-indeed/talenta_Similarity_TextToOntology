from typing import Union, List

import sys
import spacy
import subprocess
import numpy as np
from .base import BaseSimilarity

# Ignore warnings of spacy
import warnings
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)

class SpacyTransformerSimilarity(BaseSimilarity):
    """
    Args:
        - model_name:str = Efficiency->'en_core_web_md', Accuracy->'en_core_web_trf'
    """
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List], model_name:str="en_core_web_md") -> None:
        super().__init__(a, b)
        try:
            self.model = spacy.load(model_name)
        except OSError as e:
            print(f"Couldn't find model {model_name} on local. Downloading and installing the model...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spacy",
                    "download",
                    model_name
                ]
            )
            self.model = spacy.load(model_name)
        
        self.similarity_matrix(self.transformers_similarity)

    def preprocess(self, a, b):
        doc_function = lambda text: self.model(str(text))
        a = list(map(doc_function, a))
        b = list(map(doc_function, b))
        return a, b

    def transformers_similarity(self, x, y):
        return x.similarity(y)