import numpy as np


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
