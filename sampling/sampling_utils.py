import networkx as nx
import os


def save_sampled_walks(G, walks, fname, dir = './'):
    """
    saved the sampled graph and the walks to local file
    :param G: the sampled graph
    :param walks: the sampled walks
    """
    print('Sampling Utils: saving sampled walks to local file ...')
    if not os.path.exists(dir):
        os.mkdir(dir)

    # save the sampled graph
    if G is not None:
        nx.write_edgelist(G, dir + fname + '.edgelist', data=['weight'])
    # save the sampled walks
    if walks is not None:
        with open(dir + fname + '.txt', 'w') as f:
            for walk in walks:
                line = ''
                for node in walk:
                    line = line + str(node) + ' '
                f.write('%s\n' % (line,))


def load_sampled_walks(fname):
    """
    read the sampled walks from file
    """
    walks = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            nodes = line.split(' ')

            walk = []
            for node in nodes:
                walk.append(int(node))
            walks.append(walk)
    return walks
