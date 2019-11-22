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
from sampling.node2vec_random_walk_sampling import Node2VecRandomWalkSampling


# noinspection PyMissingConstructor
class DfsWalkSampling(StaticClassSampling):
    """ We approximate DFS sampling by setting p to a big value and q to a small value in node2vec sampling """

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

        self.is_directed = is_direct
        self.num_walks_iter = kwargs['num_walks_iter']
        self.max_sampled_walk = kwargs['max_sampled_walk']
        self.walk_length = kwargs['walk_length']
        self.p = 4
        self.q = 0.25
        kwargs['p'] = self.p
        kwargs['q'] = self.q
        kwargs['node2vec_c_executable'] = 'node2vec'
        kwargs['is_use_python'] = False

        self.name = 'approximate-dfs-walk'
        self.sampling_model = Node2VecRandomWalkSampling(G, edge_file, is_direct, **kwargs)

    def get_description(self):
        return {'name': self.get_name(), 'walk_length': self.walk_length, 'num_walks_iter': self.num_walks_iter, 'p': self.p, 'q': self.q}

    def get_name(self):
        return self.name

    def get_sampled_graph(self):
        return self.sampling_model.get_sampled_graph()


def run_test():
    pass


if __name__ == '__main__':
    run_test()
