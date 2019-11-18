import importlib
import numpy as np
import time
import platform
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

import matplotlib.pyplot as plt
import networkx as nx
import os
import importlib
import platform

from sampling.node2vec_random_walk_sampling import Node2VecRandomWalkSampling

from embedding import embedding_utils

sampling = None
sampling_utils = None


glove = None
Glove = None
Corpus = None
# noinspection PyBroadException
try:
    glove = importlib.import_module('glove')
except Exception:
    print('Failed to import glove - system info: ' + platform.platform())
if glove is not None:
    Glove = getattr(glove, 'Glove')
    Corpus = getattr(glove, 'Corpus')


class GloveEmbedding:

    def __init__(self, d, **kwargs):
        """
        The initializer of the Node2VecEmbedding class
        :param kwargs: a dict contains:
            d: dimension of the embedding
            window_size: context size for optimization
            max_iter: max number of iterations
            n_workers: number of parallel workers
        """
        self._method_name = 'GloVe-Embedding'
        self.d = d
        self.max_iter = kwargs['max_iter']
        self.walks = kwargs['walks']
        self.window_size = kwargs['window_size']
        self.n_workers = kwargs['n_workers']
        self.learning_rate = kwargs['learning_rate']
        self.verbose = kwargs['verbose']
        self.embedding = None
        self._node_num = None
    
    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self.d)

    def get_embedding(self):
        return self.embedding

    def get_edge_weight(self, i, j):
        return np.dot(self.embedding[i, :], self.embedding[j, :])

    def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
        t1 = time.time()
        walks = self.walks

        # crete the Corpus from the walks
        walks = [list(map(str, walk)) for walk in walks]
        corpus_model = Corpus()
        corpus_model.fit(walks, window=self.window_size)

        dictionary = corpus_model.dictionary
        print('Dict size: %s' % len(dictionary))
        self._node_num = len(corpus_model.dictionary)
        print('Collocations: %s' % corpus_model.matrix.nnz)

        # train the glove model
        glove_model = Glove(no_components=self.d, learning_rate=self.learning_rate)
        glove_model.fit(corpus_model.matrix, epochs=int(self.max_iter), no_threads=self.n_workers, verbose=self.verbose)
        glove_model.add_dictionary(dictionary)

        # generate the embedding
        word_vectors = glove_model.word_vectors
        embedding = np.zeros((self._node_num, self.d))
        for node_i in range(0, self._node_num):
            node_word_id = dictionary[str(node_i)]
            embedding[node_i, :] = word_vectors[node_word_id]
        self.embedding = embedding

        t2 = time.time()

        return self.embedding, t2-t1

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


def run_experiment(data_path, sampled_walk_file=None, is_save_walks=False):
    print("Starting experiment ...")
    # use random walk to sample from the graph
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    is_directed = False

    if sampled_walk_file is not None:
        sampled_graph = nx.read_edgelist(data_path, data=(('weight', float),), create_using=nx.Graph, nodetype=int)
        walks = sampling_utils.load_sampled_walks(sampled_walk_file)
    else:
        random_walk_sampling = get_node2vec_random_walk_sampling(data_path, is_directed)
        sampled_graph, walks = random_walk_sampling.get_sampled_graph()
        # save to local file
        if is_save_walks:
            fname = random_walk_sampling.get_name() + '-' + str(time.time())
            walks_dir = './sampled_walks/'
            if not os.path.exists(walks_dir):
                os.mkdir(walks_dir)
            walks_dir = walks_dir + data_name + '/'
            sampling_utils.save_sampled_walks(G=None, walks=walks, dir=walks_dir, fname=fname)

    print('number of nodes in the sampled graph: ', sampled_graph.number_of_nodes())
    print('number of edges in the sampled graph: ', sampled_graph.number_of_edges())
    print('number of walks: ', len(walks))
    print('walk length: ', len(walks[0]))
    # make the sampled graph into directed graph as in GEM
    sampled_graph = sampled_graph.to_directed()
    # we can also save the sampled graph and the walks to file at the end

    # generate embedding
    emb_dir = '../output/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    emb_dir += (data_name + '/')
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)

    models = list()
    if glove is not None:
        models.append(get_glove_model(walks))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    print('\n\nStart learning embedding ...')
    print('Num nodes: %d, num edges: %d' % (sampled_graph.number_of_nodes(), sampled_graph.number_of_edges()))
    for embedding in models:
        print('\nLearning embedding using %s ...' % (embedding.get_method_name(),))
        t1 = time.time()
        # Learn embedding - accepts a networkx graph or file with edge list
        learned_embedding, t = embedding.learn_embedding(graph=sampled_graph, edge_f=None, is_weighted=True, no_python=True)
        # Save embedding to file
        embedding_utils.save_embedding_to_file(learned_embedding, emb_dir + data_name + '_' + embedding.get_method_name() + '.emb')
        print(embedding.get_method_name() + ':\n\tTraining time: %f' % (time.time() - t1))


def get_node2vec_random_walk_sampling(data_path, is_directed):
    kwargs = dict()
    kwargs['p'] = 1
    kwargs['q'] = 1
    kwargs['walk_length'] = 80  # default value: 80
    # the default algorithm samples num_walks_iter walks starting for each node
    kwargs['num_walks_iter'] = 10
    # set the maximum number of sampled walks (if None, the algorithm will sample from the entire graph)
    kwargs['max_sampled_walk'] = None

    return Node2VecRandomWalkSampling(None, data_path, is_directed, **kwargs)


def get_glove_model(walks):
    kwargs = dict()
    d = 128
    kwargs['max_iter'] = 5
    kwargs['walks'] = walks
    kwargs['window_size'] = 10
    kwargs['n_workers'] = 4
    kwargs['learning_rate'] = 0.05
    kwargs['verbose'] = False
    return GloveEmbedding(d, **kwargs)


if __name__ == '__main__':
    data_list = ['../data/blog-catalog-deepwalk/blog-catalog.edgelist']
    sampled_walks_list = ['../sampled_walks/blog-catalog/node2vec-random-walk-1573955197.082777.txt']
    is_save_walks_list = [True]

    for i in range(0, len(data_list)):
        print('Run experiment using dataset: ' + data_list[i])
        if sampled_walks_list[i] is not None:
            print('Run experiment using sampled walks: ' + str(sampled_walks_list[i]))
        run_experiment(data_list[i], sampled_walks_list[i], is_save_walks_list[i])
