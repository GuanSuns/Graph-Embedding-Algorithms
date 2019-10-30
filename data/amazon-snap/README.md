### The Amazon Product Co-Purchasing Network and Ground-Truth Communities in SNAP
- **raw data**: http://snap.stanford.edu/data/com-Amazon.html
- **content**: 334863 nodes, 925872 links
- **preprocessor/parser**: [amazon_snap_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/amazon_snap_preprocessor.py)
- **amazon-snap.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
- **amazon-snap-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in amazon-snap-top5000-community-info.txt
- **amazon-snap-top5000-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in amazon-snap-community-info.txt
- **citation**: J. Yang and J. Leskovec. Defining and Evaluating Network Communities based on Ground-truth. ICDM, 2012.
