import matplotlib.pyplot as plt
from time import time
import networkx as nx
import os
from datetime import datetime
import importlib
import platform

from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr

from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne import SDNE
from argparse import ArgumentParser

from sampling.node2vec_random_walk_sampling import Node2VecRandomWalkSampling
from embedding.node2vec_embedding import Node2VecEmbedding
from embedding.fast_text_embedding import FastTextEmbedding
from embedding.glove_embedding import GloveEmbedding
from embedding.cbow_embedding import CBOWEmbedding
from embedding import embedding_utils
from sampling import sampling_utils

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


def run_experiment(data_path, sampled_walk_file=None, is_save_walks=False):
    print("Starting experiment ...")
    # use random walk to sample from the graph
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    is_directed = False

    if sampled_walk_file is not None:
        sampled_graph = nx.read_edgelist(data_path, data=(('weight', float),), create_using=nx.Graph(), nodetype=int)
        walks = sampling_utils.load_sampled_walks(sampled_walk_file)
    else:
        random_walk_sampling = get_node2vec_random_walk_sampling(data_path, is_directed)
        sampled_graph, walks = random_walk_sampling.get_sampled_graph()
        # save to local file
        if is_save_walks:
            fname = random_walk_sampling.get_name() + '-' + str(datetime.timestamp(datetime.now()))
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
    emb_dir = 'output/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    emb_dir += (data_name + '/')
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    # Choose from ['GraphFactorization', 'HOPE', 'LaplacianEigenmaps'
    # , 'LocallyLinearEmbedding', 'node2vec', 'FastText', 'CBOW', 'Glove']
    model_to_run = ['node2vec', 'FastText', 'CBOW', 'LocallyLinearEmbedding']
    models = list()

    # Load the models you want to run
    if 'node2vec' in model_to_run:
        models.append(get_node2vec_model(walks))
    if 'FastText' in model_to_run:
        models.append(get_fast_text_model(walks))
    if 'CBOW' in model_to_run:
        models.append(get_cbow_model(walks))
    if 'Glove' in model_to_run and glove is not None:
        models.append(get_glove_model(walks))
    if 'GraphFactorization' in model_to_run:
        models.append(GraphFactorization(d=128, max_iter=1000, eta=1 * 10 ** -4, regu=1.0))
    if 'HOPE' in model_to_run:
        models.append(HOPE(d=128, beta=0.01))
    if 'LaplacianEigenmaps' in model_to_run:
        models.append(LaplacianEigenmaps(d=128))
    if 'LocallyLinearEmbedding' in model_to_run:
        models.append(LocallyLinearEmbedding(d=128))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    print('\n\nStart learning embedding ...')
    print('Num nodes: %d, num edges: %d' % (sampled_graph.number_of_nodes(), sampled_graph.number_of_edges()))
    for embedding in models:
        print('\nLearning embedding using %s ...' % (embedding.get_method_name(), ))
        t1 = time()
        # Learn embedding - accepts a networkx graph or file with edge list
        learned_embedding, t = embedding.learn_embedding(graph=sampled_graph, edge_f=None, is_weighted=True, no_python=True)
        # Save embedding to file
        embedding_utils.save_embedding_to_file(learned_embedding, emb_dir + data_name + '_' + embedding.get_method_name() + '_' + str(t1) + '.emb')
        print(embedding.get_method_name() + ':\n\tTraining time: %f' % (time() - t1))
        # you can add visualization and evaluation here
        # Example: Evaluate on graph reconstruction
        # MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(sampled_graph, embedding, learned_embedding, None)
        # ---------------------------------------------------------------------------------
        # print(("\tMAP: {} \t precision curve: {}\n\n\n\n" + '-' * 100).format(MAP, prec_curv[:5]))


def get_node2vec_random_walk_sampling(data_path, is_directed):
    kwargs = dict()
    kwargs['p'] = 1
    kwargs['q'] = 1
    kwargs['walk_length'] = 80  # default value: 80
    # the default algorithm samples num_walks_iter walks starting for each node
    kwargs['num_walks_iter'] = 10
    # set the maximum number of sampled walks (if None, the algorithm will sample from the entire graph)
    kwargs['max_sampled_walk'] = None
    # use c executable as default
    kwargs['is_use_python'] = False
    kwargs['node2vec_c_executable'] = 'node2vec'

    return Node2VecRandomWalkSampling(None, data_path, is_directed, **kwargs)


def get_node2vec_model(walks):
    kwargs = dict()
    d = 128
    kwargs['max_iter'] = 1
    kwargs['walks'] = walks
    kwargs['window_size'] = 10
    kwargs['n_workers'] = 5

    return Node2VecEmbedding(d, **kwargs)


def get_fast_text_model(walks):
    kwargs = dict()
    d = 128
    kwargs['max_iter'] = 1
    kwargs['walks'] = walks
    kwargs['window_size'] = 10
    kwargs['n_workers'] = 5

    return FastTextEmbedding(d, **kwargs)


def get_cbow_model(walks):
    kwargs = dict()
    d = 128
    kwargs['max_iter'] = 1
    kwargs['walks'] = walks
    kwargs['window_size'] = 10
    kwargs['n_workers'] = 5
    return CBOWEmbedding(d, **kwargs)


def get_glove_model(walks):
    kwargs = dict()
    d = 128
    kwargs['max_iter'] = 1
    kwargs['walks'] = walks
    kwargs['window_size'] = 10
    kwargs['n_workers'] = 5
    kwargs['learning_rate'] = 0.05
    return GloveEmbedding(d, **kwargs)


if __name__ == '__main__':
    # candidate: ['./data/blog-catalog-deepwalk/blog-catalog.edgelist', './data/flickr-deepwalk/flickr-deepwalk.edgelist', './data/sbm/sbm.edgelist']
    data_list = ['./data/sbm/sbm.edgelist']
    # candidate: ['./sampled_walks/blog-catalog/node2vec-random-walk-1574042236.322876.txt', './sampled_walks/flickr-deepwalk/node2vec-random-walk-1574063574.331607.txt']
    sampled_walks_list = [None]
    is_save_walks_list = [True]

    for i in range(0, len(data_list)):
        print('Run experiment using dataset: ' + data_list[i])
        if sampled_walks_list[i] is not None:
            print('Run experiment using sampled walks: ' + str(sampled_walks_list[i]))
        run_experiment(data_list[i], sampled_walks_list[i], is_save_walks_list[i])
