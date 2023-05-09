# indeed-similarity
A pipeline to find similarity for 2 list of text

Please see the usage example [here](examples/similairty.ipynb).

## Description
The indeed-similarity needs two inputs which are two list of strings. It give the output as a dictionary containing all specified functions including average. 
The similarity score of all functions ranges from 0 (the lowest similarity) and 1 (the highest similarity). However, there might be some case that text similarity score of transformer-based model can be negative.

Please note that you can always add a new similarity function by having the BaseSimilarity and the superclass.

## Similarity Function

1. Jaccard: [ref](https://www.educative.io/answers/what-is-the-jaccard-similarity-measure-in-nlp#)
  Jaccard similarity is nothing but the score of intersection of two input strings. In each loop, the algorithm feeds the pairs of string from two inputs into the Jaccard similarity to find the ratio.

2. Levenshtein: [ref](https://maxbachmann.github.io/Levenshtein/levenshtein.html)
  In each loop, the algorithm feeds the pairs of string from two inputs into the levenshtein.ratio function to find the normalized distance.

3. Difflib: [ref](https://docs.python.org/3/library/difflib.html)
  In each loop, the algorithm feeds the pairs of string from two inputs into the a function from diff lib called SequenceMatcher to find the similarity score. SequenceMatcher is the algorithm that will loop through of words/phases with four different operator (replace, delete, insert, and equal) in order to turn from string1 into string2. Then it will calculate the similarity score accordingly.
  For example, the inputs are “My name is Mike” and “Hi”. First, we need to replace “My name “ from the first string with “H”. Then do nothing with “i”. Finally, “s Mike” is removed.



| Operator 	| Item (1 --> 2)     	| String (1 --> 2)   	|
|----------	|--------------------	|--------------------	|
| replace  	| a[0:8] --> b[0:1]  	| 'My name ' --> 'H' 	|
| equal    	| a[8:9] --> b[1:2]  	| 'i' --> 'i'        	|
| delete   	| a[9:15] --> b[2:2] 	| ''s Mike' --> ''   	|


4. S-Bert: [ref](https://www.sbert.net/)
  S-Bert is a transformer based model. It is described that the model can be useful for semantic textual similar, semantic search, or paraphrase mining. In each loop, a string from first list and a string from second list are encoded (vectorised) using a specified model from S-Bert. Then cosine similarity is used to find the text similarity between two encoded strings.

5. Spacy: [ref](https://spacy.io/usage/linguistic-features)
  Spacy is similar to S-Bert because it is also a transformer based model. In each loop, a string from first list and a string from second list are encoded (vectorised) using a specified model from Spacy. Then cosine similarity (by default) is used to find the text similarity between two encoded strings.
