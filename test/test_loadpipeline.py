from indeed_similarity.similarity import Pipeline, LevenshteinSimilarity

def test_loadpipeline():
    Pipeline([LevenshteinSimilarity])