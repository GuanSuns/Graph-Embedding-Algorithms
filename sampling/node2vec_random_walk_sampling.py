# node2vec_random_walk_sampling.py
# -----------------------------------------
# Implement the random walk sampling used in Node2Vec
# The implementation is based on Aditya Grover's code: https://github.com/aditya-grover/node2vec
import os
import networkx as nx
import sys
import numpy as np
import random
import time

from sampling.static_graph_sampling import StaticClassSampling
from sampling import sampling_utils
from subprocess import call


# noinspection PyMissingConstructor
class Node2VecRandomWalkSampling(StaticClassSampling):

    def __init__(self, G, edge_file, is_direct, **kwargs):
        """

        :param G: the networkx graph
        :param sampled_size: the number of edges to be included in the sampled graph
        :param edge_file: when G is none, it will read the edgelist file
        :param kwargs: args should be a dict which includes 'p', 'q', 'walk_length'
                    ,'num_walks_iter', 'max_sampled_walk'
                    , 'is_use_python', 'node2vec_c_executable'
        """
        if G is None:
            if is_direct:
                G = nx.read_edgelist(edge_file, data=(('weight', float),), create_using=nx.DiGraph, nodetype=int)
            else:
                G = nx.read_edgelist(edge_file, data=(('weight', float),), create_using=nx.Graph, nodetype=int)

        self.G = G
        self.is_weighted = self.is_graph_weighted()
        self.edge_file = edge_file
        self.is_directed = is_direct
        self.num_walks_iter = kwargs['num_walks_iter']
        self.max_sampled_walk = kwargs['max_sampled_walk']
        self.p = kwargs['p']
        self.q = kwargs['q']
        self.walk_length = kwargs['walk_length']
        # use python implementation or C implementation
        self.is_use_python = True
        if 'is_use_python' in kwargs:
            self.is_use_python = kwargs['is_use_python']
        # the command to run node2vec using C executable from snap
        self.node2vec_c_executable = None
        if 'node2vec_c_executable' in kwargs:
            self.node2vec_c_executable = kwargs['node2vec_c_executable']

        # variables used in functions
        self.alias_nodes = None
        self.alias_edges = None

        self.name = 'node2vec-random-walk'

    def get_name(self):
        return self.name

    def is_graph_weighted(self):
        G = self.G
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                if 'weight' not in G[node][neighbor]:
                    return False
                else:
                    return True

    def get_sampled_graph(self):
        if not self.is_use_python:
            walks = self.simulate_walks_c_executable()
            if walks is not None:
                return self.G, walks

        self.preprocess_transition_probs()
        return self.simulate_walks(self.walk_length, self.num_walks_iter, self.max_sampled_walk)

    def simulate_walks_c_executable(self):
        temp_file = 'temp_walks-' + str(time.time()) + '.txt'
        # noinspection PyListCreation
        args = [self.node2vec_c_executable]
        args.append('-i:' + self.edge_file)
        args.append('-o:' + temp_file)
        args.append('-l:%d' % self.walk_length)
        args.append('-r:%d' % self.num_walks_iter)
        args.append('-p:%f' % self.p)
        args.append('-q:%f' % self.q)
        if self.is_directed:
            args.append('-dr')
        if self.is_weighted:
            args.append('-w')
        args.append('-ow')
        args.append('-v')

        try:
            print(args)
            call(args)
        except Exception as e:
            print(str(e))
            print(self.node2vec_c_executable + ' not found. Please compile snap, place node2vec in the system path and grant executable permission')
            return None

        # read the walks from the file
        walks = sampling_utils.load_sampled_walks(temp_file)
        os.remove(temp_file)

        return walks

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
        print("\nTotal number of nodes: ", len(nodes))
        print("Total number of edges: ", n_edges)

        print("Start random walks ...")
        walk_iter = 0
        n_sampled_walk = 0
        is_stopped = False
        while not is_stopped:
            random.shuffle(nodes)
            for i_node, node in enumerate(nodes):
                sys.stdout.write('\r')
                sys.stdout.write('Sampling from node %d' % (i_node,))
                sys.stdout.flush()

                sampled_walk = self.node2vec_walk(walk_length=walk_length, start_node=node)
                walks.append(sampled_walk)

                previous_node = sampled_walk[0]
                # use the sampled_walk to construct the sampled_graph
                for i in range(1, len(sampled_walk)):
                    current_node = sampled_walk[i]
                    if current_node != previous_node:
                        sampled_graph.add_edge(previous_node, current_node, weight=self.get_edge_weight(previous_node, current_node))
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

        print('\nFinish sampling ...')
        return sampled_graph, walks

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev_node = walk[-2]
                    next_node = cur_nbrs[alias_draw(alias_edges[(prev_node, cur)][0], alias_edges[(prev_node, cur)][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(self.get_edge_weight(dst, dst_nbr)/ p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.get_edge_weight(dst, dst_nbr))
            else:
                unnormalized_probs.append(self.get_edge_weight(dst, dst_nbr) / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Pre-processing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        print('Node2Vec Random Walk preprocessing probs ...')
        print('\tStart processing nodes: ', G.number_of_nodes())

        alias_nodes = {}
        for i_node, node in enumerate(G.nodes()):
            sys.stdout.write('\r')
            sys.stdout.write('\tProcessing node %d' % (i_node,))
            sys.stdout.flush()

            unnormalized_probs = [self.get_edge_weight(node, nbr) for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        print('\n\tStart processing edges: ', G.number_of_edges())
        if is_directed:
            for i_edge, edge in enumerate(G.edges()):
                sys.stdout.write('\r')
                sys.stdout.write('\tProcessing edge %d' % (i_edge,))
                sys.stdout.flush()
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for i_edge, edge in enumerate(G.edges()):
                sys.stdout.write('\r')
                sys.stdout.write('\tProcessing edge %d' % (i_edge,))
                sys.stdout.flush()
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

    def get_edge_weight(self, node, nbr):
        G = self.G
        if 'weight' in G[node][nbr]:
            return G[node][nbr]['weight']
        else:
            return 1.0


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def run_test():
    data_path = '../data/karate/karate.edgelist'
    is_directed = False

    kwargs = dict()
    kwargs['p'] = 0.25
    kwargs['q'] = 0.25
    kwargs['walk_length'] = 80
    # the default algorithm samples num_walks_iter walks starting for each node
    kwargs['num_walks_iter'] = 10
    # set the maximum number of sampled walks (if None, the algorithm will sample from the entire graph)
    kwargs['max_sampled_walk'] = None
    kwargs['is_use_python'] = False
    kwargs['node2vec_c_executable'] = 'node2vec'

    node2vec_random_walk_sampling = Node2VecRandomWalkSampling(None, data_path, is_directed, **kwargs)
    sampled_graph, walks = node2vec_random_walk_sampling.get_sampled_graph()
    print('number of nodes in the sampled graph: ', sampled_graph.number_of_nodes())
    print('number of edges in the sampled graph: ', sampled_graph.number_of_edges())
    print('number of walks: ', len(walks))
    print('walk length: ', len(walks[0]))

    return sampled_graph, walks


if __name__ == '__main__':
    run_test()
