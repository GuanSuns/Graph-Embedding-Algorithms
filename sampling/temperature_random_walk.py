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
class TemperatureRandomWalkSampling(StaticClassSampling):

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
        self.max_sampled_walk = kwargs['max_sampled_walk']
        self.walk_length = kwargs['walk_length']

        # parameter related to the logistic function: n(t) = L/(1+ math.exp(-k*(t-t0)))
        self.t0 = kwargs['t0']      # default: walk_length/2=40
        self.L = kwargs['L']
        self.k = kwargs['k']    # default: 0.1
        self.p = kwargs['p']

        self.name = 'temperature-random-walk'

    def get_description(self):
        return {'name': self.get_name(), 'L': self.L, 'k': self.k, 'p': self.p, 'walk_length': self.walk_length, 'num_walks_iter': self.num_walks_iter}

    def get_name(self):
        return self.name

    def get_sampled_graph(self):
        return self.simulate_walks(self.walk_length, self.num_walks_iter, self.max_sampled_walk)

    def simulate_walks(self, walk_length, max_walk_iteration, max_sampled_walk=None):
        """
        Repeatedly simulate random walks from each node.
        """
        if self.is_directed:
            sampled_graph = nx.DiGraph()
        else:
            sampled_graph = nx.Graph()

        walks = []
        nodes = list(self.G.nodes())
        n_edges = self.G.number_of_edges()
        print("Total number of nodes: ", len(nodes))
        print("Total number of edges: ", n_edges)

        print("Start random walks ...")
        walk_iter = 0
        n_sampled_walk = 0
        is_stopped = False
        while not is_stopped:
            random.shuffle(nodes)
            for node in nodes:
                sys.stdout.write('\r')
                sys.stdout.write('Walk iteration: %d' % (n_sampled_walk,))
                sys.stdout.flush()
                # print("- Walk iteration: ", str(walk_iter + 1))
                sampled_walk = self.simple_random_walk(walk_length=walk_length, start_node=node)
                walks.append(sampled_walk)
                # print(sampled_walk)
                previous_node = sampled_walk[0]
                # use the sampled_walk to construct the sampled_graph
                for i in range(1, len(sampled_walk)):
                    current_node = sampled_walk[i]
                    if current_node != previous_node:
                        sampled_graph.add_edge(previous_node, current_node)
                    # update previous node
                    previous_node = current_node
                # count sampled walk
                n_sampled_walk += 1
                if max_sampled_walk is not None and n_sampled_walk >= max_sampled_walk:
                    is_stopped = True
                    break

            walk_iter += 1
            if walk_iter >= max_walk_iteration:
                is_stopped = True
                break

        return sampled_graph, walks

    def simple_random_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        L = self.L
        k = self.k
        t0 = self.t0
        init_p = self.p

        walk = [start_node]
        t = 0
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))

            if len(walk) == 1:
                next_node = np.random.choice(cur_nbrs)
                walk.append(next_node)
                continue

            cur_nbrs.append(cur)
            pre_node = walk[-2]
            pre_neighbors = list(G.neighbors(pre_node))
            probs = TemperatureRandomWalkSampling.get_transition_probs(pre_neighbors, pre_node, cur_nbrs, L, k, t0, t, init_p)
            probs = probs/np.sum(probs)
            next_node = np.random.choice(cur_nbrs, p=probs)
            walk.append(next_node)
            t += 1
        return walk

    @staticmethod
    def get_transition_probs(pre_neighbors, pre_node, candidate, L, k, t0, t, init_p):
        current_p = TemperatureRandomWalkSampling.get_logistic_value(L, k, t0, t, init_p)
        probs = np.zeros(shape=(len(candidate), ))
        for i, node in enumerate(candidate):
            if node == pre_node:
                probs[i] = 1/current_p
            elif node in pre_neighbors:
                probs[i] = 1
            else:
                probs[i] = current_p
        return probs

    @staticmethod
    def get_logistic_value(L, k, t0, t, init_p):
        return min(L/(1 + math.exp(- k * (t - t0))) + init_p, 1/init_p)


def run_test():
    pass


if __name__ == '__main__':
    run_test()
