import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

alpha_mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
                 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w',
                 23: 'x', 24: 'y', 25: 'z'}


def get_random_graph(seed: int, *, min_node_count: int = 2, max_node_count: int = 10,
                     set_labels: bool = False, allow_self_loops: bool = False) -> nx.Graph:
    """
    seed: random seed
    min_node_count / max_node_count: defines upper and lower bounds of graph size
    set_labels: flag to optionally define own labeling of nodes
    allow_self_loops: allow self loops as edges
    return: networkx graph
    """

    np.random.seed(seed)
    node_count = np.random.randint(min_node_count, max_node_count)
    adj = np.random.randint(0, 2, (node_count, node_count))  # Generate random square binary matrix
    if not allow_self_loops:
        np.fill_diagonal(adj, 0)
    G = nx.from_numpy_matrix(adj)
    if set_labels:
        nx.relabel_nodes(G, alpha_mapping, copy=False)

    return G


if __name__ == '__main__':
    graph = get_random_graph(123146)
    nx.draw_networkx(graph, node_size=800)
    plt.show()
