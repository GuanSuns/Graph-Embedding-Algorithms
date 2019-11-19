"""
The abstract base class of different sampling methods
"""
from abc import ABCMeta
import networkx as nx


class StaticClassSampling:
    __metaclass__ = ABCMeta

    def __init__(self, G, edge_file, is_directed, sampled_size, **kwargs):
        """ Initialize the sampling class

        :param G: the networkx graph
        :param sampled_size: the number of edges to be included in the sampled graph
        :param is_directed: the graph is directed or not
        :param edge_file: when G is none, it will read the edgelist file
        :param **kwargs: other parameters required to initialized the sampler
        """
        pass

    def get_sampled_graph(self):
        """
        :return: a sampled sub-graph
        """
        return nx.Graph()




