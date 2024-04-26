# Text To Ontology
The Text_To_Ontology.py file is mainly focused on processing text documents and generating ontologies out of said documents. Several python libraries such as spaCy, Natural Language Toolkit (nltk) and torch are used for their Natural Language Processing capabilities.

# Overall Script Procedure
1. Library imports
    1. Import required Python library for analysis & processing
2. (Pre-processing) Load Pre-trained Model
    1. spaCy - advanced natural language processing
3. Input Text Modification
    1. Text extraction from file
    2. Text split into sentences
    3. Sentences cleaned of punctuation, numbers, symbols
4. (Ontology) Loading Ontologies
    1. Selection & Usage of specific ontology for procedure
5. Further work
    1. Annotation of text
    2. Embeddings generated using Bert & fastText models
6. Lamba function that calculates cosine similarity between embeddings is invoked
7. Comparison of Models
    1. Results are plotted on graph
    2. Performances of Bert & fastText are compared