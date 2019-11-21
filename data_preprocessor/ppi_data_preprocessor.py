import scipy.io
import numpy as np
import sys
import networkx as nx


def load_graph(data_fname, label_fname, is_directed=False):
    if is_directed:
        original_G = nx.read_edgelist(data_fname, data=(('weight', float),), create_using=nx.DiGraph(), nodetype=int)
        G = nx.DiGraph()
    else:
        original_G = nx.read_edgelist(data_fname, data=(('weight', float),), create_using=nx.Graph(), nodetype=int)
        G = nx.Graph()

    # construct the graph
    print('Reading graph file ...')
    node2id = dict()
    next_available_id = 0
    for node in list(original_G.nodes()):
        node_id = node
        if node not in node2id:
            node2id[node] = next_available_id
            node_id = next_available_id
            next_available_id += 1

        for neighbor in original_G.neighbors(node):
            neighbor_id = neighbor
            if neighbor not in node2id:
                node2id[neighbor] = next_available_id
                neighbor_id = next_available_id
                next_available_id += 1

            G.add_edge(int(node_id), int(neighbor_id), weight=original_G[node][neighbor]['weight'])

    n_node = original_G.number_of_nodes()
    print('\nNumber of nodes: %d, number of edges: %d' % (n_node, original_G.number_of_edges()))
    print('Number of nodes in node2id dict: ', len(node2id))

    # reading label files
    node_info = {}
    node_labels = [[] for _ in range(0, n_node)]
    n_labels = None
    with open(label_fname, 'r') as f:
        for i, line in enumerate(f):
            # show the processing progress
            sys.stdout.write('\r')
            sys.stdout.write('Processing line %d' % (i,))
            sys.stdout.flush()

            if line[0] == '#':
                continue
            labels = line.strip().split()
            if n_labels is None:
                n_labels = len(labels) - 1

            node = int(labels[0])
            node_id = node2id[node]

            for l in range(0, n_labels):
                if int(labels[l+1]) == 1:
                    node_labels[node_id].append(l)

    node_info['label'] = node_labels
    return G, node_info


def save_graph_to_edge_list(graph, label_info, fname):
    # save the graph, format: node_from, node_to, weight
    print('\nWriting graph into edgelist file ...')
    edge_list_fname = fname + '.edgelist'
    with open(edge_list_fname, 'w') as f:
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('%d %d %f\n' % (i, j, w))

    print('\nWriting label into txt file ...')
    label_fname = fname + '-label.txt'
    with open(label_fname, 'w') as f:
        for node_id in range(0, len(label_info)):
            line = str(node_id)
            for l in range(0, len(label_info[node_id])):
                line += ' '
                line += str(l)
            f.write('%s\n' % (line,))


def main():
    graph_file = '../raw_data/PPI/PPI.edgelist'
    label_file = '../raw_data/PPI/PPI.edgelist.truth'
    graph, node_info = load_graph(graph_file, label_file)
    # save the graph into edgelist
    # save the labels to txt, format: node_id [list of labels]
    save_graph_to_edge_list(graph, node_info['label'], fname='ppi')


if __name__ == '__main__':
    main()


