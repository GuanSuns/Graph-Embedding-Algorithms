from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gensim.models import Word2Vec
import numpy as np
import time

from sampling import node2vec_random_walk_sampling
from embedding import embedding_utils


# noinspection PyMissingConstructor
class CBOWEmbedding(StaticGraphEmbedding):

    def __init__(self, d, **kwargs):
        """
        The initializer of the Node2VecEmbedding class
        :param kwargs: a dict contains:
            d: dimension of the embedding
            window_size: context size for optimization
            max_iter: max number of iterations
            n_workers: number of parallel workers
        """
        self._method_name = 'CBOW-Embedding'
        self.d = d
        self.max_iter = kwargs['max_iter']
        self.walks = kwargs['walks']
        self.num_walks = len(self.walks)
        self.walk_len = len(self.walks[0])
        self.window_size = kwargs['window_size']
        self.n_workers = kwargs['n_workers']
        self.embedding = None
        self._node_num = None

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
        # sg = 1 means SkipGram otherwise CBOW
        model = Word2Vec(walks, size=self.d, window=self.window_size, min_count=0
                         , sg=0, workers=self.n_workers, iter=self.max_iter)
        self.embedding = embedding_utils.gensim_model_to_embedding(model, walks)
        self._node_num = self.embedding.shape[0]
        t2 = time.time()
        return self.embedding, t2-t1

    def get_embedding(self):
        return self.embedding

    def get_edge_weight(self, i, j):
        return np.dot(self.embedding[i, :], self.embedding[j, :])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self.embedding = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


def run_test():
    # use random walk to sample from the graph
    sampled_graph, walks = node2vec_random_walk_sampling.run_test()

    # create embedding
    kwargs = dict()
    d = 4
    kwargs['walks'] = walks
    kwargs['max_iter'] = 1
    kwargs['window_size'] = 5
    kwargs['n_workers'] = 1

    CBOW_embedding = CBOWEmbedding(d, **kwargs)
    CBOW_embedding.learn_embedding(graph=sampled_graph, edge_f=None, is_weighted=True, no_python=True)


if __name__ == '__main__':
    run_test()
