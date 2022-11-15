import json

import networkx as nx
import numpy as np


def from_adj_str(adj_s: str, node_labels=None) -> nx.Graph:
    """
    adj_s: adjacency matrix of graph in form of string
    node_labels: optional node labeling flag
    return: networkx graph
    """

    adj = np.array(json.loads(adj_s))
    if len(adj.shape) == 2 and adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix has to be 2D and square. Was {adj.shape}")
    graph = nx.from_numpy_matrix(adj)
    if node_labels:
        nx.relabel_nodes(graph, {i: node_labels[i] for i in range(len(node_labels))})
    return graph


if __name__ == '__main__':
    s = """
    [[0,1,0],
    [0,0,1],
    [1,0,0]]"""
    labels = ["a", "b", "c"]
    g = from_adj_str(s, labels)
    print(g)
    print(g.nodes)
