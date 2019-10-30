import scipy.io
import numpy as np
import sys
import networkx as nx


def load_graph_from_mat(data_path):
    # Load the data
    print('\nReading dataset ...')
    data_mat = scipy.io.loadmat(data_path)
    label_array = data_mat['group'].toarray()
    network_sparse_matrix = data_mat['network']

    n_label = label_array.shape[1]
    n_node = label_array.shape[0]

    # iterate through the network to construct the graph
    print('Creating graph from scipy sparse matrix ...')
    graph = nx.from_scipy_sparse_matrix(network_sparse_matrix)
    node_labels = [[] for _ in range(0, n_node)]
    node_info = dict()

    cnt_node_with_label = 0
    for i_node in range(0, n_node):
        # show the processing progress
        sys.stdout.write('\r')
        sys.stdout.write('Processing node %d' % (i_node, ))
        sys.stdout.flush()

        has_label = False
        for l in range(0, n_label):
            if int(label_array[i_node][l]) == 1:
                node_labels[i_node].append(l)
                has_label = True

        if has_label:
            cnt_node_with_label += 1

        # save the label info into the graph
        graph.nodes[i_node]['labels'] = node_labels[i_node]

    print('\nProportion of nodes with label info: %.2f%%' % (float(cnt_node_with_label)/n_node*100.0, ))
    node_info['labels'] = node_labels

    return graph, node_info


def save_graph_to_edge_list(graph, node_labels, fname='youtube-deepwalk'):
    # save the graph, format: node_from, node_to, weight
    print('\nWriting graph into edgelist file ...')
    edge_list_fname = fname + '.edgelist'
    with open(edge_list_fname, 'w') as f:
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('%d %d %f\n' % (i, j, w))

    # save the node-labels to txt, format: node_id labels
    print('Writing node_labels into txt file ...')
    node_labels_fname = fname + '-labels.txt'
    with open(node_labels_fname, 'w') as f:

        for i_node in range(0, len(node_labels)):
            line = str(i_node)
            if len(node_labels[i_node]) == 0:
                continue

            for l in range(0, len(node_labels[i_node])):
                line += ' '
                line += str(node_labels[i_node][l])
            f.write('%s\n' % (line, ))


def main():
    data_youtube_file = '../raw_data/youtube-deepwalk/youtube.mat'
    youtube_graph, youtube_node_info = load_graph_from_mat(data_youtube_file)
    save_graph_to_edge_list(youtube_graph, youtube_node_info['labels'])


if __name__ == '__main__':
    main()
