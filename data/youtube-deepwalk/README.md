## The YouTube dataset in DeepWalk
- **raw data**: http://leitang.net/social_dimension.html
- **content**: 1138499 nodes, 2990443 links and 47 categories.
- **preprocessor/parser**: [youtube_deepwalk_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/youtube_deepwalk_preprocessor.py)
- **youtube-deepwalk-edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
- **youtube-deepwalk-labels**: each line represents a node and the labels it has; the format is (node_id, [list of labels]); it should be noticed that not all the node in this dataset have been assigned a label (only about 3% of nodes have at least one label).
- **citation**: L. Tang and H. Liu. Scalable learning of collective behavior based on sparse social dimensions. In Proceedings of the 18th ACM conference on Information and knowledge management, pages 1107–1116. ACM, 2009.
 