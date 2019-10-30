import scipy.io
import numpy as np
import sys
import networkx as nx


def load_graph_from_txt(data_fname, community_fname, community_top5000_fname=None, is_directed=False):
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # construct the graph
    print('Reading graph file ...')
    with open(data_fname, 'r') as f:
        for i, line in enumerate(f):
            # show the processing progress
            sys.stdout.write('\r')
            sys.stdout.write('Processing line %d' % (i,))
            sys.stdout.flush()

            if line[0] == '#':
                continue
            edge = line.strip().split()
            if len(edge) == 3:
                w = float(edge[2])
            else:
                w = 1.0
            G.add_edge(int(edge[0]), int(edge[1]), weight=1)

    n_node = G.number_of_nodes()
    print('\nNumber of nodes: %d, number of edges: %d' % (n_node, G.number_of_edges()))

    node_info = {'community_info': None, 'top5000_community_info': None}
    community_info = dict()
    top5000_community_info = dict()

    # parse community info
    print('Reading community file ...')
    if community_fname is not None:
        with open(community_fname, 'r') as f:
            for community_id, line in enumerate(f):
                # show the processing progress
                sys.stdout.write('\r')
                sys.stdout.write('Processing community %d' % (community_id,))
                sys.stdout.flush()

                nodes = line.strip().split()
                # add community id to each node
                for node in nodes:
                    node_id = int(node)
                    if node_id not in community_info:
                        community_info[node_id] = []
                    community_info[node_id].append(community_id)

        node_info['community_info'] = community_info

    # parse top 5000 community info
    print('\nReading top 5000 community file ...')
    if community_top5000_fname is not None:
        with open(community_top5000_fname, 'r') as f:
            for community_id, line in enumerate(f):
                # show the processing progress
                sys.stdout.write('\r')
                sys.stdout.write('Processing community %d' % (community_id,))
                sys.stdout.flush()

                nodes = line.strip().split()
                # add community id to each node
                for node in nodes:
                    node_id = int(node)
                    if node_id not in top5000_community_info:
                        top5000_community_info[node_id] = []
                    top5000_community_info[node_id].append(community_id)

        node_info['top5000_community_info'] = top5000_community_info

    return G, node_info


def save_graph_to_edge_list(graph, node_info, fname):
    # save the graph, format: node_from, node_to, weight
    print('\nWriting graph into edgelist file ...')
    edge_list_fname = fname + '.edgelist'
    with open(edge_list_fname, 'w') as f:
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('%d %d %f\n' % (i, j, w))

    # save the community_info to txt, format: node_id [list of community it belongs to]
    if node_info['community_info'] is not None or len(node_info['community_info']) > 0:
        print('Writing community_info into txt file ...')
        community_info_fname = fname + '-community-info.txt'
        community_info = node_info['community_info']
        write_community_info_to_txt(community_info, community_info_fname)

    # save the top 5000 community_info to txt, format: node_id [list of community it belongs to]
    if node_info['top5000_community_info'] is not None or len(node_info['top5000_community_info']) > 0:
        print('Writing top5000_community_info into txt file ...')
        top5000_community_info_fname = fname + '-top5000-community-info.txt'
        top5000_community_info = node_info['top5000_community_info']
        write_community_info_to_txt(top5000_community_info, top5000_community_info_fname)


def write_community_info_to_txt(community_info, community_info_fname):
    with open(community_info_fname, 'w') as f:
        for i_node, node_community_list in community_info.items():
            line = str(i_node)
            if len(community_info[i_node]) == 0:
                continue

            for community_index in range(0, len(node_community_list)):
                line += ' '
                line += str(node_community_list[community_index])
            f.write('%s\n' % (line,))

