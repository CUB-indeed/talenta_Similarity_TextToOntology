# Text To Ontology
The Text_To_Ontology.py file is mainly focused on processing text documents and generating ontologies out of said documents. Several python libraries such as spaCy, Natural Language Toolkit (nltk) and torch are used for their Natural Language Processing capabilities.

# Overall Script Procedure
1. Library imports
- Import required Python library for analysis & processing
2. (Pre-processing) Load Pre-trained Model
- spaCy - advanced natural language processing
3. Input Text Modification
- Text extraction from file
- Text split into sentences
- Sentences cleaned of punctuation, numbers, symbols
4. (Ontology) Loading Ontologies
- Selection & Usage of specific ontology for procedure
5. Further work
- Annotation of text
- Embeddings generated using Bert & fastText models
6. Lamba function that calculates cosine similarity between embeddings is invoked
7. Comparison of Models
- Results are plotted on graph
- Performances of Bert & fastText are compared