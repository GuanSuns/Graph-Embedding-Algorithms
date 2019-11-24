# Graph-Embedding-Algorithms


## The Pipeline

- **To generate word embedding**
    - File: **run\_experiment.py**
    - In the **main** function, set the dataset (or the walks files) and other options
    - In the **run\_experiment** function, choose the sampling strategy and the word embedding models to be used

- **To evaluate the word embedding**
    - File: **run\_evaluation.py**
    - Results: the results will be saved in the evaluation\_results.csv file
    - In the **run\_evaluation** function, set the embedding files to be used
    - In the **run\_classification\_experiment** function, set the multi-label classifiers to be used
    
- **Notes**
    - The pipeline is inherited from the GEM library, in which we can add evaluation modules in GEM
        - Example from GEM: https://github.com/palash1992/GEM/blob/master/examples/run_karate.py
    
## Sampling Strategies

- **Biased Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/biased_walk.py
    - **citation**: Nguyen, Duong, and Fragkiskos D. Malliaros. "BiasedWalk: Biased Sampling for Representation Learning on Graphs." 2018 IEEE International Conference on Big Data (Big Data). IEEE, 2018.
    - **note**: the implementation is based on the author's [open-source code](https://github.com/duong18/BiasedWalk/tree/master/source)

- **Node2Vec Biased Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/node2vec_random_walk_sampling.py
    - **citation**: Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016.
    - **note**: the implementation is based on the author's [open-source code](https://github.com/aditya-grover/node2vec)

- **Uniform Simple Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/simple_random_walk_sampling.py
    - **note**: the simple random walk which uniformly chooses the neighbor to visit

- **Approximated BFS Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/bfs_walk_sampling.py
    - **note**: the approximated BFS random walk by setting p = 0.25 and q = 4 in node2vec

- **Approximated DFS Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/temperature_random_walk.py
    - **note**: use logistic function to make the random walk gradually switch from BFS to DFS

- **The Combined Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/dfs_walk_sampling.py
    - **note**: the approximated BFS random walk by setting p = 4 and q = 0.25 in node2vec


## Word Embedding Methods

- **CBOW**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/embedding/cbow_embedding.py
    - **note**: use the word2vec embedding model in gensim
    - **citation**: Rehurek, Radim, and Petr Sojka. "Software framework for topic modelling with large corpora." In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. 2010.

- **SkipGram**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/embedding/node2vec_embedding.py
    - **note**: the SkipGram word embedding used in node2vec
    - **citation**: Rehurek, Radim, and Petr Sojka. "Software framework for topic modelling with large corpora." In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. 2010.

- **Fast-Text**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/embedding/fast_text_embedding.py
    - **note**: the Fast-Text word embedding in gensim
    - **citation**: Rehurek, Radim, and Petr Sojka. "Software framework for topic modelling with large corpora." In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. 2010.

- **Glove**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/embedding/glove_embedding.py
    - **note**: the Glove word embedding in glove_python
    - **source**: https://github.com/maciejkula/glove-python
    - **citation**: Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.


## Dataset
All the dataset in the data directory have been parsed into edgelist format, which are compatible with the library [GEM: Graph Embedding Methods](https://github.com/palash1992/GEM) and the [node2vec (python implementation)](https://github.com/aditya-grover/node2vec).

### Due to file size limitation on Github, you might need to parse and generate some of the large dataset by yourself from the original raw dataset. The corresponding preprocessor/parser functions can be found under the [data_preprocessor directory](https://github.com/GuanSuns/Graph-Embedding-Algorithms/tree/master/data_preprocessor) 


- **The Flickr dataset in DeepWalk**
    - **raw data**: http://leitang.net/social_dimension.html
    - **content**: 80513 nodes, 5899882 links, and 195 categories.
    - **preprocessor/parser**: [flickr_deepwalk_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/flickr_deepwalk_preprocessor.py)
    - **flickr-deepwalk.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **flickr-deepwalk-labels.txt**: each line represents a node and the labels it has; the format is (node_id, [list of labels]).
    - **citation**: L. Tang and H. Liu. Relational learning via latent social dimensions. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’09, pages 817–826, New York, NY, USA, 2009. ACM.
    

- **The YouTube dataset in DeepWalk**
    - **raw data**: http://leitang.net/social_dimension.html
    - **content**: 1138499 nodes, 2990443 links and 47 categories.
    - **preprocessor/parser**: [youtube_deepwalk_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/youtube_deepwalk_preprocessor.py)
    - **youtube-deepwalk.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **youtube-deepwalk-labels.txt**: each line represents a node and the labels it has; the format is (node_id, [list of labels]); it should be noticed that not all the node in this dataset have been assigned a label (only about 3% of nodes have at least one label).
    - **citation**: L. Tang and H. Liu. Scalable learning of collective behavior based on sparse social dimensions. In Proceedings of the 18th ACM conference on Information and knowledge management, pages 1107–1116. ACM, 2009.


- **The BlogCatalog dataset in DeepWalk**
    - **raw data**: http://leitang.net/social_dimension.html
    - **content**: 10312 nodes, 333983 links, and 39 categories.
    - **preprocessor/parser**: [blog_catalog_deepwalk_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/blog_catalog_deepwalk_preprocessor.py)
    - **blog-catalog.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **blog-catalog-labels.txt**: each line represents a node and the labels it has; the format is (node_id, [list of labels]).
    - **citation**: L. Tang and H. Liu. Relational learning via latent social dimensions. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’09, pages 817–826, New York, NY, USA, 2009. ACM.

- **The Amazon Product Co-Purchasing Network and Ground-Truth Communities in SNAP**
    - **raw data**: http://snap.stanford.edu/data/com-Amazon.html
    - **content**: 334863 nodes, 925872 links
    - **preprocessor/parser**: [amazon_snap_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/amazon_snap_preprocessor.py)
    - **amazon-snap.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **amazon-snap-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in amazon-snap-top5000-community-info.txt
    - **amazon-snap-top5000-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in amazon-snap-community-info.txt
    - **citation**: J. Yang and J. Leskovec. Defining and Evaluating Network Communities based on Ground-truth. ICDM, 2012.
    
- **Youtube Social Network and Ground-Truth Communities in SNAP**
    - **raw data**: http://snap.stanford.edu/data/com-Youtube.html
    - **content**: 1134890 nodes, 2987624 links
    - **preprocessor/parser**: [youtube_snap_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/youtube_snap_preprocessor.py)
    - **youtube-snap.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **youtube-snap-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in youtube-snap-top5000-community-info.txt
    - **youtube-snap-top5000-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in youtube-snap-community-info.txt
    - **citation**: J. Yang and J. Leskovec. Defining and Evaluating Network Communities based on Ground-truth. ICDM, 2012.
    
- **LiveJournal Social Network and Ground-Truth Communities in SNAP**
    - **raw data**: http://snap.stanford.edu/data/com-LiveJournal.html
    - **content**: 3997962 nodes, 34681189 links
    - **preprocessor/parser**: [live_journal_snap_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/live_journal_snap_preprocessor.py)
    - **live-journal-snap.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **live-journal-snap-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in live-journal-snap-top5000-community-info.txt
    - **live-journal-snap-top5000-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in live-journal-snap-community-info.txt
    - **citation**: J. Yang and J. Leskovec. Defining and Evaluating Network Communities based on Ground-truth. ICDM, 2012.

- **The Protein-Protein Interaction (PPI) dataset** 
    - **raw data**: http://jiongqianliang.com/MILE/
    - **content**: 3852 nodes, 38705 links
    - **preprocessor/parser**: [ppi_data_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/ppi_data_preprocessor.py)
    - **ppi.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **ppi-label.txt**: each line represents a node and the labels it has; the format is (node_id, [list of labels]).
    - **citation**: Breitkreutz, Bobby-Joe, et al. "The BioGRID interaction database: 2008 update." Nucleic acids research (2007)