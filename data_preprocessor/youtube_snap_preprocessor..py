from data_preprocessor.snap_community_preprocessor import load_graph_from_txt, save_graph_to_edge_list


def main():
    graph_file = '../raw_data/youtube-snap/com-youtube.ungraph.txt'
    community_file = '../raw_data/youtube-snap/com-youtube.all.cmty.txt'
    top5000_community_file = '../raw_data/youtube-snap/com-youtube.top5000.cmty.txt'
    graph, node_info = load_graph_from_txt(graph_file, community_file, top5000_community_file)
    # save the graph into edgelist
    # save the top community_info to txt, format: node_id [list of community it belongs to]
    save_graph_to_edge_list(graph, node_info, fname='youtube-snap')


if __name__ == '__main__':
    main()

