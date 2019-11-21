import numpy as np
import os
import csv


def gensim_model_to_embedding(model, walks):
    """
    Generate the embedding matrix (numpy array) from GENSIM model
    :param model: the gensim model
    :param walks: the sampled walks (in the form of [list of ints, list of ints, ...]
    """
    # find node id set
    nodes_set = set()
    for walk in walks:
        for node in walk:
            nodes_set.add(node)

    # create the embedding list
    embedding = np.zeros((len(nodes_set), int(model.vector_size)))
    for node in nodes_set:
        embedding[int(node), :] = model.wv[str(node)]

    return embedding


def save_embedding_to_file(embedding, fname, emb_description=None):
    """
    Save the embeddings to local file
    :param embedding: matrix (or 2d-array) with the format (n_node, dimension)
    :param fname: file to store embedding
    :param emb_description: a dictionary containing related information of the embedding;
                        if this is not None, save the experiment experiment_log.csv
    """
    n_node = embedding.shape[0]
    d = embedding.shape[1]

    with open(fname, 'w') as f:
        # write header (meta data)
        line = str(n_node) + ' ' + str(d)
        f.write('%s\n' % (line,))

        for node_id in range(0, n_node):
            line = str(node_id)
            for d_index in range(0, d):
                line += ' '
                line += str(embedding[node_id][d_index])
            f.write('%s\n' % (line, ))

    current_dir = os.path.dirname(fname) + '/'
    log_file = current_dir + 'experiment_log.txt'
    if emb_description is not None:
        if not os.path.exists(log_file):
            f = open(log_file, 'w')
        else:
            f = open(log_file, 'a')
        f.write('%s\n' % (str(emb_description),))
        f.close()




