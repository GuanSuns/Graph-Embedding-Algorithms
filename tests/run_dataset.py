'''
Run the graph embedding methods on Karate graph and evaluate them on
graph reconstruction and visualization.
'''
import matplotlib.pyplot as plt
from time import time

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


def try_dataset(data_path, is_directed):
    """
    try each dataset
    """
    # Load graph
    G = graph_util.loadGraphFromEdgeListTxt(data_path, directed=is_directed)
    G = G.to_directed()

    # List the models to try in the list.
    # Choose from ['GraphFactorization', 'HOPE', 'LaplacianEigenmaps', 'LocallyLinearEmbedding', 'node2vec']
    model_to_try = ['HOPE', 'node2vec']
    models = list()
    # Load the models you want to run
    if 'GraphFactorization' in model_to_try:
        models.append(GraphFactorization(d=2, max_iter=50000, eta=1 * 10 ** -4, regu=1.0))
    if 'HOPE' in model_to_try:
        models.append(HOPE(d=4, beta=0.01))
    if 'LaplacianEigenmaps' in model_to_try:
        models.append(LaplacianEigenmaps(d=2))
    if 'LocallyLinearEmbedding' in model_to_try:
        models.append(LocallyLinearEmbedding(d=2))
    if 'node2vec' in model_to_try:
        models.append(node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    for embedding in models:
        print('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        t1 = time()
        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
        # Evaluate on graph reconstruction
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
        # ---------------------------------------------------------------------------------
        print(("\tMAP: {} \t precision curve: {}\n\n\n\n" + '-' * 100).format(MAP, prec_curv[:5]))
        # ---------------------------------------------------------------------------------
        # Visualize
        viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
        plt.show()
        plt.clf()


def main():
    data_path = '../data/karate.edgelist'
    is_directed = True
    try_dataset(data_path, is_directed)


if __name__ == '__main__':
    main()
