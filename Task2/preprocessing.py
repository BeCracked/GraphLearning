from typing import Iterable
import numpy as np
import networkx as nx
import torch
from scipy.sparse import coo_matrix
from torch import tensor, Tensor


def norm_adj_matrix(g: nx.Graph, pad_length: None | int = 0) -> coo_matrix:
    """
    Computes the normalized adjacency matrix for graph g with 1/sqrt(d_i*d_j) if (v_i,v_j) \\in E or i=j.

    Parameters
    ----------
    g The graph of which to compute the normalized adjacency matrix.
    pad_length The length to pad the adjacency matrix to. Ignored if len(g.nodes) >= pad_length or pad_length = None.

    Returns
    -------
    The normalized adjacency matrix as a dense numpy array with shape (len(g.nodes), len(g.nodes)).
    """
    # Construct vector of degree+1 of all nodes
    degree_vec = np.array([val + 1 for (node, val) in g.degree()], dtype=int, ndmin=2)

    # Incidence matrix with diag set to one
    # e[i][j] = 1 iff (v_i,v_j) in E or i=j
    # Note: Casting go lil for construction (setting the diagonal) and casting back should be faster
    e = nx.adjacency_matrix(g, None, dtype=int).tolil()
    e.setdiag(1)
    # Matrix of d_i*d_j
    d_m = np.matmul(degree_vec.transpose(), degree_vec)
    # Normalized adjacency matrix
    na_m: coo_matrix = e.multiply(d_m).sqrt()
    # Take reciprocal by modifying data directly
    na_m.data = np.reciprocal(na_m.data)

    # Add padding
    if pad_length and g.number_of_nodes() < pad_length:
        na_m.resize(pad_length, pad_length)
    return na_m


def norm_adj_matrices(graphs: list[nx.Graph] | np.ndarray[nx.Graph], dtype=torch.float32) -> Tensor:
    """
    Construct the normalized adjacency matrices for all given graphs and return them as a tensor.

    Parameters
    ----------
    graphs List of graphs for which to construct the normalized adjacency matrices.
    dtype The dtype to use for the tensor. Defaults to torch.float32.

    Returns
    -------
    A 3D tensor of the shape (len(graphs), max(g.number_of_nodes), max(g.number_of_nodes)) containing the normalized
    adjacency matrices along the first dimension.
    """
    pad_length = max([g.number_of_nodes() for g in graphs])
    shape = (len(graphs), pad_length, pad_length)

    t_m = torch.zeros(shape)
    for i in range(len(graphs)):
        a = norm_adj_matrix(graphs[i], pad_length)
        t = tensor(a.toarray(), dtype=dtype)
        t_m[i] = t

    return t_m


def get_node_feature_embedding(graph: nx.Graph, features_key: str = "node_attributes",
                               node_pad_length: int = None, feature_pad_length: int = None) -> Tensor:
    """
    Constructs the tensor of shape (node_count, feature_length)
    containing the node feature vectors of the nodes for the given graph.
    ----------
    graph The graph to get the node feature embedding for.
    features_key The key where the feature data is stored in the node data dict. Defaults to "node_attributes".
    node_pad_length The length to pad the nodes (aka. number of feature vectors) to.
    feature_pad_length The length to pad the feature vectors to.

    Returns
    -------
    The tensor with the concatenated node feature vectors.
    """
    node_length = node_pad_length if node_pad_length else graph.number_of_nodes()
    feature_length = feature_pad_length if feature_pad_length else max(len(attributes) for n, attributes in graph)
    shape = (node_length, feature_length)

    t_m = torch.zeros(shape)
    nodes = list(graph.nodes(data=features_key))
    for i in range(graph.number_of_nodes()):
        node, attributes = nodes[i]
        t_m[i] = tensor(attributes)

    return t_m


def get_node_feature_embeddings(graphs: list[nx.Graph] | np.ndarray[nx.Graph],
                                features_key: str = "node_attributes") -> Tensor:
    """
    Constructs the 3D tensor of shape (len(graphs), max_number_of_nodes, max_feature_vec_length)
    containing the concatenated node feature vectors for all given graphs.

    Parameters
    ----------
    graphs List of graphs to get the node feature embeddings for.
    features_key The key where the feature data is stored in the node data dict. Defaults to "node_attributes".

    Returns
    -------
    A 3D tensor of shape (len(graphs), max_number_of_nodes, max_feature_vec_length)
    containing the concatenated node feature vectors for all given graphs.
    """
    # Length of the longest node feature vector
    max_number_of_nodes = max([g.number_of_nodes() for g in graphs])
    max_feature_vec_length = len(graphs[0].nodes(data=features_key)[1])  # Assumes all feature vectors are of same length
    shape = (len(graphs), max_number_of_nodes, max_feature_vec_length)

    t_m = torch.zeros(shape)
    for i in range(len(graphs)):
        t = get_node_feature_embedding(graphs[i], features_key,
                                       node_pad_length=max_number_of_nodes, feature_pad_length=max_feature_vec_length)
        t_m[i] = t

    return t_m


def extract_labels_from_dataset(dataset: Iterable, label_key: str = "label") -> list:
    """
    Gets the labels of a dataset for datasets that are subscript-able.

    Parameters
    ----------
    dataset The dataset of which to get the labels.
    label_key The key to look at to get the label.

    Returns
    -------
    The list of labels extracted from the dataset.
    """
    return [element.graph[label_key] for element in dataset]


if __name__ == '__main__':
    from Task1.helper.graph_gen import get_random_graph

    for k in range(20):
        G = get_random_graph(12345 + k)
        m = norm_adj_matrix(G, 15)
        print(len(G.nodes))
        print(m.toarray())
        print("####################################")
