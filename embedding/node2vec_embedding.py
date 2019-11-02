from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gensim.models import Word2Vec


# noinspection PyMissingConstructor
class Node2VecEmbedding(StaticGraphEmbedding):

    def __init__(self, d, **kwargs):
        """
        The initializer of the Node2VecEmbedding class
        :param kwargs: a dict contains:
            d: dimension of the embedding
            max_iter: max iterations
            window_size: context size for optimization
            n_workers: number of parallel workers
            ret_p: return weight
            inout_p: inout weight
        """
        self._method_name = 'Node2Vec-Embedding'
        self.d = d
        self.max_iter = kwargs['max_iter']
        self.con_size = kwargs['con_size']
        self.walks = kwargs['walks']
        self.num_walks = len(self.walks)
        self.walk_len = len(self.walks[0])
        self.window_size = kwargs['window_size']
        self.n_workers = kwargs['n_workers']
        self.embedding = None

        pass

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
        walks = self.walks
        walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks, size=self.d, window=self.window_size, min_count=0
                         , sg=1, workers=self.n_workers, iter=self.max_iter)


