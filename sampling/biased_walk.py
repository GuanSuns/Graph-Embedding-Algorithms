# simple_random_walk_sampling.py
# -----------------------------------------
# Implement the simple random walk sampling (uniformly choose the neighbor)
# This class can be used as a base class for more complicated random walk sampling

import networkx as nx
import numpy as np
import random
import sys
import os
from sampling.static_graph_sampling import StaticClassSampling


# noinspection PyMissingConstructor
class BiasedWalk(StaticClassSampling):

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
        self.i_value = kwargs['i_value']  # the initialization value of phenomenon
        self.is_bfs = kwargs['is_bfs']
        if self.is_bfs:
            self.name = 'biased-random-walk-bfs'
        else:
            self.name = 'biased-random-walk-dfs'

        additions = [1.0]
        for i in range(0, self.walk_length):
            additions.append(self.i_value * additions[i])
        self.additions = additions

    def get_description(self):
        return {'name': self.get_name(), 'walk_length': self.walk_length, 'num_walks_iter': self.num_walks_iter,
                'is_bfs': self.is_bfs, 'i_value': self.i_value}

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
                in_adj[node] = set(G.predecessors(node)).union(set(G.successors(node)))
                out_adj[node] = set(G.successors(node))
            else:
                in_adj[node] = set(G.neighbors(node))
                out_adj[node] = set(in_adj[node])

        random.shuffle(nodes)  # shuffle node set before doing random walks
        for i in range(0, max_walk_iteration):
            step = 0
            for node in nodes:
                sys.stdout.write('\r')
                sys.stdout.write('Iteration %d, step %d' % (i, step))
                sys.stdout.flush()
                step += 1

                tau = {}
                walk = [node]
                walk_l = 0

                while walk_l < walk_length - 1:
                    u = walk[walk_l]
                    if out_adj[u] == set():  # cannot go further
                        break
                    for w in in_adj[u]:  # defuse to all (in, out) neighbors
                        self.update_value(tau, w, walk_l)

                    total = 0.0
                    r = random.random()
                    temp_sum = 0.0
                    v = list(G.neighbors(u))[0]
                    if self.is_bfs:
                        for v in out_adj[u]:
                            total = total + tau[v] * self.get_edge_weight(u, v)

                        for v in out_adj[u]:
                            temp_sum = temp_sum + tau[v] * self.get_edge_weight(u, v)
                            if temp_sum / total > r:
                                break
                    else:
                        for v in out_adj[u]:
                            total = total + np.reciprocal(tau[v]) * self.get_edge_weight(u, v)

                        for v in out_adj[u]:
                            temp_sum = temp_sum + np.reciprocal(tau[v]) * self.get_edge_weight(u, v)
                            if temp_sum / total > r:
                                break

                    walk.append(v)
                    walk_l = walk_l + 1
                walks.append(walk)

        return G, walks

    def update_value(self, tau, u, l):
        if u in tau:
            tau[u] = tau[u] + self.additions[l]
        else:
            tau[u] = self.additions[l]

        return tau[u]

    @staticmethod
    def get_value(tau, u):
        if u not in tau:
            # tau[u] = self.additions[l]
            print('unexpected error')
            exit()
        return tau[u]

    def get_edge_weight(self, node, nbr):
        G = self.G
        if 'weight' in G[node][nbr]:
            return G[node][nbr]['weight']
        else:
            return 1.0


def run_test():
    pass


if __name__ == '__main__':
    run_test()
