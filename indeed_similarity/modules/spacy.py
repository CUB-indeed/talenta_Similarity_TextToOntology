from typing import Union, List, Tuple

import sys
import spacy
import subprocess
import numpy as np
from .base import BaseSimilarity

# Ignore warnings of spacy
import warnings
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)

class SpacyTransformerSimilarity(BaseSimilarity):
    def __init__(self, a:Union[np.array, List], b:Union[np.array, List], model_name:str="en_core_web_md") -> None:
        """A transformer based neural network for text similarity between two lists on Spacy package. 

        Args:
            a (Union[np.array, List]): The first list containing strings.
            b (Union[np.array, List]): The second list containing strings.
            model_name (str, optional): A pretrained model name of text simlarity in Spcay Transformer. Defaults to "en_core_web_md".
                Please see the linke [https://spacy.io/models] for more detail.
                In short, Efficiency->'en_core_web_md', Accuracy->'en_core_web_trf'
        """
        super().__init__(a, b)
        # Check whether the specified pretrained model is in the environment. If not, script will download it automatically.
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
        self()

    def preprocess(self, a:Union[np.array, List], b:Union[np.array, List]) -> Tuple[List, List]:
        doc_function = lambda text: self.model(str(text))
        a = list(map(doc_function, a))
        b = list(map(doc_function, b))
        return a, b

    def similarity_func(self, x:str, y:str) -> float:
        return x.similarity(y)