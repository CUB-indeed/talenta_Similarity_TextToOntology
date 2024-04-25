# Text To Ontology
The Text_To_Ontology.py file is mainly focused on processing text documents and generating ontologies out of said documents. Several python libraries such as spaCy, Natural Language Toolkit (nltk) and torch are used for their Natural Language Processing capabilities.

# Overall Script Procedure
1. Library imports
    1.1. Import required Python library for analysis & processing
2. (Pre-processing) Load Pre-trained Model
    2.1. spaCy - advanced natural language processing
3. Input Text Modification
    3.1. Text extraction from file
    3.2. Text split into sentences
    3.3. Sentences cleaned of punctuation, numbers, symbols
4. (Ontology) Loading Ontologies
    4.1. Selection & Usage of specific ontology for procedure
5. Further work
    5.1. Annotation of text
    5.2. Embeddings generated using Bert & fastText models
6. Lamba function that calculates cosine similarity between embeddings is invoked
7. Comparison of Models
    7.1. Results are plotted on graph
    7.2. Performances of Bert & fastText are compared

# Mason
MAnufacturing Semantics ONtology, or MASON for short, is an ontology that aims to provide a common semantic framework that supports various applications within the manufacturing sector. 
The ontology utilizes Web Ontology Language, or OWL, a Semantic Web language designed to represent rich and complex knowledge about things, groups of things, and relations between things. OWL is well-suited for creating detailed and complex ontologies in manufacturing systems. 
MASON can be classed as an upper ontology. This implies that it provides a high-level schema designed to cover broad concepts across multiple domains or applications, particularly useful for integrating and aligning more specialized or domain-specific ontologies.
### *Implementation process*
The implementation process of the MASON ontology within the Text_To_Ontology.py file is: