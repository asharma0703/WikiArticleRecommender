# WikiArticleRecommender

Contains the code for my SEC11 original research project, a Wikipedia article recommender system using network structure for recommendations.
To use, get the dependencies (numpy/scipy, tqdm) and [download the wiki-topcats dataset from SNAP](https://snap.stanford.edu/data/wiki-topcats.html).
For the input I recommend writing a file with the inputs and using `cat fname.input - | python set_similarity.py` so the program doesn't get fed EOF at the end of the file. Then you can keep playing with more inputs. Alternatively just pickle a subgraph instead of building a random one each time and it should run pretty quick.
