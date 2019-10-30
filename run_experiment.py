import networkx as nx


def main():
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)

    G.add_edge(1, 2, weight=3)
    G.add_edge(2, 3)

    for line in nx.generate_edgelist(G, data=False):
        print(line)


if __name__ == '__main__':
    main()
