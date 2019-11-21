### The Flickr dataset in DeepWalk
- **raw data**: http://leitang.net/social_dimension.html
- **content**: 80513 nodes, 5899882 links, and 195 categories.
- **preprocessor/parser**: [flickr_deepwalk_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/flickr_deepwalk_preprocessor.py)
- **flickr-deepwalk-edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
- **flickr-deepwalk-labels**: each line represents a node and the labels it has; the format is (node_id, [list of labels]).
- **citation**: L. Tang and H. Liu. Relational learning via latent social dimensions. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’09, pages 817–826, New York, NY, USA, 2009. ACM.