import networkx as nx
import os


def save_sampled_walks(G, walks, fname, dir = './', walk_description=None):
    """
    saved the sampled graph and the walks to local file
    :param G: the sampled graph
    :param walks: the sampled walks
    :param walk_description: a dictionary containing related information of the walk;
                        if this is not None, save the experiment experiment_log.csv
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

    current_dir = os.path.dirname(dir + fname + '.txt') + '/'
    log_file = current_dir + 'experiment_log.txt'
    if walk_description is not None:
        if not os.path.exists(log_file):
            f = open(log_file, 'w')
        else:
            f = open(log_file, 'a')
        f.write('%s\n' % (str(walk_description), ))
        f.close()


def load_sampled_walks(fname):
    """
    read the sampled walks from file
    """
    walks = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            nodes = line.split(' ')
            if len(nodes) == 0 or nodes[0] == '':
                print('loading sampled walks: empty line')
                continue

            walk = []
            for node in nodes:
                walk.append(int(node))
            walks.append(walk)
    return walks
