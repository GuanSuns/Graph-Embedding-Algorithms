### Youtube Social Network and Ground-Truth Communities in SNAP
- **raw data**: http://snap.stanford.edu/data/com-Youtube.html
- **content**: 1134890 nodes, 2987624 links
- **preprocessor/parser**: [youtube_snap_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/youtube_snap_preprocessor.py)
- **youtube-snap.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
- **youtube-snap-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in youtube-snap-top5000-community-info.txt
- **youtube-snap-top5000-community-info.txt**: each line represents a node and the top5000 community it belongs to; the format is (node_id, [list of community_id]). It should be noticed that the community id is not the same as the community id in youtube-snap-community-info.txt
- **citation**: J. Yang and J. Leskovec. Defining and Evaluating Network Communities based on Ground-truth. ICDM, 2012.
