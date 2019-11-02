from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gensim.models import Word2Vec
import time

from sampling.node2vec_random_walk_sampling import Node2VecRandomWalkSampling


# noinspection PyMissingConstructor
class Node2VecEmbedding(StaticGraphEmbedding):

    def __init__(self, d, **kwargs):
        """
        The initializer of the Node2VecEmbedding class
        :param kwargs: a dict contains:
            d: dimension of the embedding
            window_size: context size for optimization
            max_iter: max number of iterations
            n_workers: number of parallel workers
        """
        self._method_name = 'Node2Vec-Embedding'
        self.d = d
        self.max_iter = kwargs['max_iter']
        self.walks = kwargs['walks']
        self.num_walks = len(self.walks)
        self.walk_len = len(self.walks[0])
        self.window_size = kwargs['window_size']
        self.n_workers = kwargs['n_workers']
        self.embedding = None

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self.d)

    def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
        """
        Return the learned embedding. This class only implements the embedding creating part
        of the node2vec, so it only takes the walks (list) in the kwargs as argument

        :param graph: won't be used in Node2VecEmbedding
        :param edge_f: won't be used in Node2VecEmbedding
        :param is_weighted: won't be used in Node2VecEmbedding
        :param no_python: won't be used in Node2VecEmbedding
        """
        t1 = time.time()
        walks = self.walks
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=self.d, window=self.window_size, min_count=0
                         , sg=1, workers=self.n_workers, iter=self.max_iter)
        t2 = time.time()
        return

def main():
    # use random walk to sample from the graph
    data_path = '../data/karate/karate.edgelist'
    is_directed = False

    kwargs = dict()
    kwargs['p'] = 0.25
    kwargs['q'] = 0.25
    kwargs['walk_length'] = 15  # default value: 80
    kwargs['num_walks_iter'] = 10  # default value: 10

    node2vec_random_walk_sampling = Node2VecRandomWalkSampling(None, data_path, is_directed, **kwargs)
    sampled_graph, walks = node2vec_random_walk_sampling.get_sampled_graph()
    print('number of nodes in the sampled graph: ', sampled_graph.number_of_nodes())
    print('number of edges in the sampled graph: ', sampled_graph.number_of_edges())
    print('number of walks: ', len(walks))
    print('walk length: ', len(walks[0]))

    # create embedding
    kwargs = dict()
    d = 4
    kwargs['walks'] = walks
    kwargs['max_iter'] = 1
    kwargs['window_size'] = 5
    kwargs['n_workers'] = 1

    node2vec_embedding = Node2VecEmbedding(d, **kwargs)
    node2vec_embedding.learn_embedding(graph=sampled_graph, edge_f=None, is_weighted=True, no_python=True)


if __name__ == '__main__':
    main()
