import networkx as nx
from typing import List


def edge_labels_to_one_hot(graphs: List[nx.Graph], *, edge_feature_key="edge_label") -> List[nx.Graph]:
    """
    Parameters
    ----------
    graphs List of networkx graphs.
    edge_feature_key Keyword to extract edge labels.
    
    Returns
    -------
    A copy of graphs with the edge labels transformed to a one hot encoding representation.
    """

    # We have edge labels 1, 2, 3
    edge_labels = list(range(1, 4))
    new_graphs = []
    for graph in graphs:
        new_labels = dict()
        for edge in graph.edges(data=True):
            new_labels[(edge[0], edge[1])] = edge[2][edge_feature_key]

        for (v, w), value in new_labels.items():
            one_hot = [0] * len(edge_labels)
            one_hot[edge_labels.index(new_labels[(v, w)])] = 1
            new_labels[(v, w)] = one_hot

        new_graph = graph.copy()
        nx.set_edge_attributes(new_graph, new_labels, name=edge_feature_key)
        new_graphs.append(new_graph)

    return new_graphs


def node_labels_to_one_hot(graphs: List[nx.Graph], *, node_feature_key="node_label") -> List[nx.Graph]:
    """
    Parameters
    ----------
    graphs List of networkx graphs.
    node_feature_key Keyword to extract node labels.
    
    Returns
    -------
    A copy of graphs with the node labels transformed to a one hot encoding representation.
    """

    # We have node labels 0 to 20
    node_labels = list(range(21))
    new_graphs = []
    for graph in graphs:
        new_labels = dict()
        for node in graph.nodes(data=True):
            new_labels[node[0]] = node[1][node_feature_key]

        for v, value in new_labels.items():
            one_hot = [0] * len(node_labels)
            one_hot[node_labels.index(new_labels[v])] = 1
            new_labels[v] = one_hot

        new_graph = graph.copy()
        nx.set_node_attributes(new_graph, new_labels, name=node_feature_key)
        new_graphs.append(new_graph)

    return new_graphs
