# simple_random_walk_sampling.py
# -----------------------------------------
# Implement the simple random walk sampling (uniformly choose the neighbor)
# This class can be used as a base class for more complicated random walk sampling

import networkx as nx
import numpy as np
import random
from sampling.static_graph_sampling import StaticClassSampling


# noinspection PyMissingConstructor
class SimpleRandomWalkSampling(StaticClassSampling):

    def __init__(self, G, edge_file, is_direct, **kwargs):
        """

        :param G: the networkx graph
        :param sampled_size: the number of edges to be included in the sampled graph
        :param edge_file: when G is none, it will read the edgelist file
        :param kwargs: args should be a dict which includes 'walk_length' ,'num_walks_iter' and 'max_sampled_walk'
        """
        if G is None:
            if is_direct:
                G = nx.read_edgelist(edge_file, data=(('weight', float),), create_using=nx.DiGraph, nodetype=int)
            else:
                G = nx.read_edgelist(edge_file, data=(('weight', float),), create_using=nx.Graph, nodetype=int)

        self.G = G
        self.is_directed = is_direct
        self.num_walks_iter = kwargs['num_walks_iter']
        self.max_sampled_walk = kwargs['max_sampled_walk']
        self.walk_length = kwargs['walk_length']

        self.name = 'sample-random-walk'

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
                # print("- Walk iteration: ", str(walk_iter + 1))
                sampled_walk = self.simple_random_walk(walk_length=walk_length, start_node=node)
                walks.append(sampled_walk)
                # print(sampled_walk)
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

        return sampled_graph, walks

    def simple_random_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                transition_probs = self.get_transition_prob(cur, cur_nbrs)
                next_node = np.random.choice(cur_nbrs, p=transition_probs)
                walk.append(next_node)
            else:
                break

        return walk

    def get_transition_prob(self, cur_node, neighbors):
        sum_weights = 0
        probs = np.zeros(shape=(len(neighbors), ))

        for i, nbr in enumerate(neighbors):
            nbr_weight = self.get_edge_weight(cur_node, nbr)
            sum_weights += nbr_weight
            probs[i] = nbr_weight
        return probs/sum_weights

    def get_edge_weight(self, node, nbr):
        G = self.G
        if 'weight' in G[node][nbr]:
            return G[node][nbr]['weight']
        else:
            return 1.0


def run_test():
    data_path = '../data/karate/karate.edgelist'
    is_directed = False

    kwargs = dict()
    kwargs['walk_length'] = 15
    # the default algorithm samples num_walks_iter walks starting for each node
    kwargs['num_walks_iter'] = 10
    # set the maximum number of sampled walks (if None, the algorithm will sample from the entire graph)
    kwargs['max_sampled_walk'] = None

    simple_random_walk_sampling = SimpleRandomWalkSampling(None, data_path, is_directed, **kwargs)
    sampled_graph, walks = simple_random_walk_sampling.get_sampled_graph()
    print('number of nodes in the sampled graph: ', sampled_graph.number_of_nodes())
    print('number of edges in the sampled graph: ', sampled_graph.number_of_edges())
    print('number of walks: ', len(walks))
    print('walk length: ', len(walks[0]))

    return sampled_graph, walks


if __name__ == '__main__':
    run_test()
