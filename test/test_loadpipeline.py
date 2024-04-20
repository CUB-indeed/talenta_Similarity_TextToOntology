from indeed_similarity.similarity import SimilarityPipeline
from indeed_similarity.modules import (
    LevenshteinSimilarity,
    JaccardSimilarity,
    SequenceSimilarity,
    BertTransformerSimilarity,
    SpacyTransformerSimilarity
)

def test_pipeline():
    SIMILARITY_FUNCTIONS = [
        LevenshteinSimilarity,
        JaccardSimilarity,
        SequenceSimilarity,
        BertTransformerSimilarity,
        SpacyTransformerSimilarity
    ]
    
    # Pre-processing functions
    preprocessing_fn = [
        lambda text: " ".join(str(text).split("_")),
        # lambda text: p.clean(str(text)),
        # lambda text: " ".join(re.findall("[A-Z][^A-Z]*", text)),
        # lambda text: re.sub(r"(?<=[a-z])\s+(?=[A-Z])", "", text),
    ]

    # Post-processing functions (for visualzation of the string in the result only)
    postprocessing_fn = [
        lambda text: "_".join(str(text).split(" ")),
    ]
    
    similarity = SimilarityPipeline(
        SIMILARITY_FUNCTIONS, preprocessing_fn, postprocessing_fn,
    )
    
    a_list_1 = ["head", "tail", "arm", "human", "finger", "feet"]
    a_list_2 = ["toe", "thumb", "hand", "headache", "armchair", "hair", "people", "finger nail", "toe_nail"]
    similarity(a_list_1, a_list_2)