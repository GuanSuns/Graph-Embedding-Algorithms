import matplotlib.pyplot as plt
from time import time
import os

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


def run_experiment():
    # use random walk to sample from the graph
    data_path = 'data/karate/karate.edgelist'
    is_directed = False

    node2vec_random_walk_sampling = get_node2vec_random_walk_sampling(data_path, is_directed)
    sampled_graph, walks = node2vec_random_walk_sampling.get_sampled_graph()
    print('number of nodes in the sampled graph: ', sampled_graph.number_of_nodes())
    print('number of edges in the sampled graph: ', sampled_graph.number_of_edges())
    print('number of walks: ', len(walks))
    print('walk length: ', len(walks[0]))
    # we can also save the sampled graph and the walks to file at the end

    # generate embedding
    emb_dir = 'output/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    # Choose from ['GraphFactorization', 'HOPE', 'LaplacianEigenmaps', 'LocallyLinearEmbedding', 'node2vec']
    model_to_run = ['node2vec']
    models = list()

    # Load the models you want to run
    if 'GraphFactorization' in model_to_run:
        models.append(GraphFactorization(d=2, max_iter=50000, eta=1 * 10 ** -4, regu=1.0))
    if 'HOPE' in model_to_run:
        models.append(HOPE(d=4, beta=0.01))
    if 'LaplacianEigenmaps' in model_to_run:
        models.append(LaplacianEigenmaps(d=2))
    if 'LocallyLinearEmbedding' in model_to_run:
        models.append(LocallyLinearEmbedding(d=2))
    if 'node2vec' in model_to_run:
        models.append(get_node2vec_model(walks))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    for embedding in models:
        print('Num nodes: %d, num edges: %d' % (sampled_graph.number_of_nodes(), sampled_graph.number_of_edges()))
        t1 = time()
        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(graph=sampled_graph, edge_f=None, is_weighted=True, no_python=True)
        print(embedding.get_method_name() + ':\n\tTraining time: %f' % (time() - t1))
        # Evaluate on graph reconstruction
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(sampled_graph, embedding, Y, None)
        # ---------------------------------------------------------------------------------
        print(("\tMAP: {} \t precision curve: {}\n\n\n\n" + '-' * 100).format(MAP, prec_curv[:5]))
        # ---------------------------------------------------------------------------------
        # Visualize
        viz.plot_embedding2D(embedding.get_embedding(), di_graph=sampled_graph, node_colors=None)
        plt.show()
        plt.clf()


def get_node2vec_random_walk_sampling(data_path, is_directed):
    kwargs = dict()
    kwargs['p'] = 0.25
    kwargs['q'] = 0.25
    kwargs['walk_length'] = 15  # default value: 80
    # the default algorithm samples num_walks_iter walks starting for each node
    kwargs['num_walks_iter'] = 10
    # set the maximum number of sampled walks (if None, the algorithm will sample from the entire graph)
    kwargs['max_sampled_walk'] = None

    return Node2VecRandomWalkSampling(None, data_path, is_directed, **kwargs)


def get_node2vec_model(walks):
    kwargs = dict()
    d = 4
    kwargs['max_iter'] = 1
    kwargs['walks'] = walks
    kwargs['window_size'] = 5
    kwargs['n_workers'] = 1

    return Node2VecEmbedding(d, **kwargs)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
