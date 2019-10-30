## The BlogCatalog dataset in DeepWalk
- **raw data**: http://leitang.net/social_dimension.html
- **content**: 10312 nodes, 333983 links, and 39 categories.
- **preprocessor/parser**: [blog_catalog_deepwalk_preprocessor](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/blog_catalog_deepwalk_preprocessor.py)
- **blog_catalog-edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
- **blog_catalog-labels**: each line represents a node and the labels it has; the format is (node_id, [list of labels]).
- **citation**: L. Tang and H. Liu. Relational learning via latent social dimensions. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’09, pages 817–826, New York, NY, USA, 2009. ACM.
