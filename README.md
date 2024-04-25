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

# Mason
MAnufacturing Semantics ONtology, or MASON for short, is an ontology that aims to provide a common semantic framework that supports various applications within the manufacturing sector. 
The ontology utilizes Web Ontology Language, or OWL, a Semantic Web language designed to represent rich and complex knowledge about things, groups of things, and relations between things. OWL is well-suited for creating detailed and complex ontologies in manufacturing systems. 
MASON can be classed as an upper ontology. This implies that it provides a high-level schema designed to cover broad concepts across multiple domains or applications, particularly useful for integrating and aligning more specialized or domain-specific ontologies.
### *Implementation process*
The implementation process of the MASON ontology within the Text_To_Ontology.py file is:
1. Introduction

The code starts by defining an empty list `mason_class`. It then iterates over each class `cls` in the `mason.classes()`. For each class, it checks if there are annotations associated with it. If annotations exist, it appends the name of the class `cls.name` to the `mason_class` list.
```python
mason_class = []

for cls in mason.classes():
    annotations = cls.comment
    if annotations:
        mason_class.append(cls.name)
```
The code then lists all classes and properties in the MASON ontology and stores them in `mason_cls` and `mason_property` lists, respectively.
```python
mason.classes()
mason_cls = list(mason.classes())

mason.properties()
mason_property = list(mason.properties())
```
For each class `ann` in `mason_cls`, the code prints the name and the comment attached to it. In the next step, the comments are appended to the list of `mason_annotations`.
```python
mason_annotations = []
for ann in mason_cls:
    print(f"\t{ann}: {ann.comment}")
    mason_annotations.append(str(ann.comment))
```
For each annotation in the list, any occurrence of a full stop is removed. Subsequently, the full stops are added back at the end of each element that can be found in the `mason_annotations` list. The code then joins all annotations into a single string `mason_ann`.
```python
# Remove the full stop from each element
mason_annotations = [x.replace('.', '') for x in mason_annotations]

for i in range(len(mason_annotations)):
    mason_annotations[i] = mason_annotations[i] + "."
```
A regular expression `regex`  is created in order to further clean up the existing `mason_ann` string. When applied to the string, `regex` performs targeted text substitutions to clean up and refine `mason_ann` further, ensuring it meets specific formatting requirements.
```python
# define a regular expression to match non-word characters except full stops
regex = re.compile('[%s]' % re.escape(string.punctuation.replace('.', '')))

# apply the regular expression to remove non-word characters
mason_ann = regex.sub('', mason_ann)
```
The code then splits `mason_ann` into sentences using periods followed by spaces as separators and stores them in `sentences_mason`. Leading & trailing whitespaces from each sentence in `sentences_mason` are then removed.
```python
sentences_mason = mason_ann.split('. ')
valid_sentences_mason = [s.strip() for s in sentences_mason]
```
A translation table is then defined to remove punctuation, symbols, and hyphens, after which it is applied to `valid_sentences_mason`. Characters such newline (\n), tab (\t), and dash (–) are also removed from each sentence.
```python
# Define a translation table to remove punctuations, symbols, and hyphens
translator_mason = str.maketrans("", "", string.punctuation + "’‘“”")

# Remove punctuations, symbols, and hyphens from each element of the list
valid_sentences_mason = [s.translate(translator_mason).strip() for s in valid_sentences_mason]

#Removing unnecessary punctuations from the text
valid_sentences_mason = [n.replace('\n', '') for n in valid_sentences_mason]
valid_sentences_mason = [n.replace('\t', '') for n in valid_sentences_mason]
valid_sentences_mason = [n.replace('–', '') for n in valid_sentences_mason]
```
The duplicates in `valid_sentences_mason` are removed through the use of the `set()` function. The `duplicates` is found and later displayed, as is the case with the total number of valid sentences.
```python
# Chexking for duplicates
unique_items = set(valid_sentences_mason)
duplicates = [item for item in unique_items if valid_sentences_mason.count(item) > 1]
```
2. Usage of Bert Model