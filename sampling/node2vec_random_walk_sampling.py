import networkx as nx
import numpy as np
import random
from sampling.static_graph_sampling import StaticClassSampling


# noinspection PyMissingConstructor
class Node2VecRandomWalkSampling(StaticClassSampling):

    def __init__(self, G, edge_file, is_direct, sampled_size, args):
        """

        :param G: the networkx graph
        :param sampled_size: the number of edges to be included in the sampled graph
        :param edge_file: when G is none, it will read the edgelist file
        :param args: args should be a dict which includes 'p', 'q', 'walk_length'
        """
        if G is None:
            if is_direct:
                G = nx.read_edgelist(edge_file, create_using=nx.DiGraph)
            else:
                G = nx.read_edgelist(edge_file, create_using=nx.Graph)

        self.G = G
        self.is_directed = is_direct
        self.sampled_size = sampled_size
        self.p = args['p']
        self.q = args['q']
        self.walk_length = args['walk_length']

        # variables used in functions
        self.alias_nodes = None
        self.alias_edges = None

    def get_sampled_graph(self):
        self.preprocess_transition_probs()
        return self.simulate_walks(self.walk_length)

    def simulate_walks(self, walk_length, max_number_walk=100000000):
        """
        Repeatedly simulate random walks from each node.
        """
        if self.is_directed:
            sampled_graph = nx.DiGraph()
        else:
            sampled_graph = nx.Graph()

        nodes = list(self.G.nodes())
        n_edges = self.G.number_of_edges()
        print("Total number of nodes: ", len(nodes))
        print("Total number of edges: ", n_edges)

        print("Walk iterations:")
        walk_iter = 0
        is_stopped = False
        while walk_iter < max_number_walk and not is_stopped:
            print("- Walk iteration: ", str(walk_iter + 1))
            random.shuffle(nodes)
            for node in nodes:
                sampled_walk = self.node2vec_walk(walk_length=walk_length, start_node=node)
                # print(sampled_walk)
                previous_node = sampled_walk[0]
                # use the sampled_walk to construct the sampled_graph
                for i in range(1, len(sampled_walk)):
                    current_node = sampled_walk[i]
                    if current_node != previous_node:
                        sampled_graph.add_edge(previous_node, current_node)
                    # update previous node
                    previous_node = current_node

                # print("\t- Current sampled edges: ", sampled_graph.number_of_edges())
                if sampled_graph.number_of_edges() >= self.sampled_size or sampled_graph.number_of_edges() == n_edges:
                    is_stopped = True
                    break

            walk_iter += 1

        return sampled_graph

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

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [self.get_edge_weight(node, nbr) for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
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


def main():
    data_path = '../data/karate/karate.edgelist'
    is_directed = False
    sampled_size = 50

    args = dict()
    args['p'] = 0.25
    args['q'] = 0.25
    args['walk_length'] = 10

    node2vec_random_walk_sampling = Node2VecRandomWalkSampling(None, data_path, is_directed, sampled_size, args)
    sampled_graph = node2vec_random_walk_sampling.get_sampled_graph()
    print('number of nodes in the sampled graph:', sampled_graph.number_of_nodes())
    print('number of edges in the sampled graph: ', sampled_graph.number_of_edges())


if __name__ == '__main__':
    main()
