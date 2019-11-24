# simple_random_walk_sampling.py
# -----------------------------------------
# Implement the simple random walk sampling (uniformly choose the neighbor)
# This class can be used as a base class for more complicated random walk sampling

import networkx as nx
import numpy as np
import random
import math
import sys
import os
from sampling.static_graph_sampling import StaticClassSampling


# noinspection PyMissingConstructor
class InfoBiasedRandomWalkSampling(StaticClassSampling):

    def __init__(self, G, edge_file, is_direct, **kwargs):
        """

        :param G: the networkx graph
        :param sampled_size: the number of edges to be included in the sampled graph
        :param edge_file: when G is none, it will read the edgelist file
        :param kwargs: args should be a dict which includes 'walk_length' ,'num_walks_iter' and 'max_sampled_walk'
        """
        if G is None:
            if is_direct:
                G = nx.read_edgelist(edge_file, data=(('weight', float),), create_using=nx.DiGraph(), nodetype=int)
            else:
                G = nx.read_edgelist(edge_file, data=(('weight', float),), create_using=nx.Graph(), nodetype=int)

        self.G = G
        self.is_directed = is_direct
        self.num_walks_iter = kwargs['num_walks_iter']
        self.walk_length = kwargs['walk_length']
        self.name = 'info-biased-random-walk'

    def get_description(self):
        return {'name': self.get_name(), 'walk_length': self.walk_length, 'num_walks_iter': self.num_walks_iter}

    def get_name(self):
        return self.name

    def get_sampled_graph(self):
        return self.simulate_walks(self.walk_length, self.num_walks_iter)

    def simulate_walks(self, walk_length, max_walk_iteration, max_sampled_walk=None):
        """
        Repeatedly simulate random walks from each node.
        """
        walks = []
        nodes = list(self.G.nodes())
        n_edges = self.G.number_of_edges()
        print("Total number of nodes: ", len(nodes))
        print("Total number of edges: ", n_edges)

        print("Start random walks ...")
        G = self.G

        in_adj = {}
        out_adj = {}  # store neighbors for each node
        for node in nodes:
            if self.is_directed:
                # note: successors and neighbors are the same for directed networks
                in_adj[node] = list(set(G.predecessors(node)).union(set(G.successors(node))))
                out_adj[node] = list(set(G.successors(node)))
            else:
                in_adj[node] = list(G.neighbors(node))
                out_adj[node] = in_adj[node]

        random.shuffle(nodes)  # shuffle node set before doing random walks
        for i in range(0, max_walk_iteration):
            step = 0
            for start_node in nodes:
                sys.stdout.write('\r')
                sys.stdout.write('Iteration %d, step %d' % (i, step))
                sys.stdout.flush()
                step += 1

                nodes_info = {}
                walk_nodes_set = set()
                walk_nodes_set.add(start_node)
                walk = [start_node]
                walk_l = 0

                while walk_l < walk_length - 1:
                    cur_node = walk[walk_l]
                    if out_adj[cur_node] == set():  # cannot go further
                        break

                    # update the info weight of its neighbors
                    neighbors = in_adj[cur_node]
                    n_neighbors = len(neighbors)
                    weights = np.zeros(shape=(n_neighbors, ))
                    for i_neighbor, neighbor in enumerate(neighbors):
                        node_info_weight = InfoBiasedRandomWalkSampling.get_info_weight(neighbor, nodes_info, in_adj, walk_nodes_set)
                        weights[i_neighbor] = node_info_weight

                    probs = weights/np.sum(weights)
                    next_node = np.random.choice(neighbors, p=probs)
                    walk.append(next_node)
                    walk_l = walk_l + 1
                    walk_nodes_set.add(next_node)

                    # update the current nodes info
                    nodes_info[cur_node] = (n_neighbors - len(set(in_adj[cur_node]) - walk_nodes_set)) / float(n_neighbors)
                walks.append(walk)

        return G, walks

    @staticmethod
    def get_info_weight(node, nodes_info, adj, walk_nodes_set):
        n_neighbor = len(adj[node])
        avg_info_weight = 0
        for neighbor in adj[node]:
            if neighbor in nodes_info:
                avg_info_weight += nodes_info[neighbor]
            else:
                d_neighbor = len(adj[neighbor])
                # didn't visit before, estimate using degree
                node_info_weight = 0.8/d_neighbor
                avg_info_weight += node_info_weight

        avg_info_weight = avg_info_weight/float(n_neighbor)
        if avg_info_weight != 0:
            avg_info_weight = 1.0/avg_info_weight

        # update nodes_info at the same time
        nodes_info[node] = (n_neighbor - len(set(adj[node]) - walk_nodes_set))/float(n_neighbor)
        return avg_info_weight


def run_test():
    pass


if __name__ == '__main__':
    run_test()
