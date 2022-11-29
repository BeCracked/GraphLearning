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


def get_node_feature_embeddings(graphs: list[nx.Graph] | np.ndarray[nx.Graph],
                                with_attributes: bool = False) -> Tensor:
    """
    Constructs the 3D tensor of shape (len(graphs), max_number_of_nodes, len(node_labels))
    containing the concatenated node feature vectors for all given graphs (in one hot encoding).

    Parameters
    ----------
    graphs List of graphs to get the node feature embeddings for.
    with_attributes Flag that considers additional node attributes if True

    Returns
    -------
    A 3D tensor of shape (len(graphs), max_number_of_nodes, len(node_labels))
    containing the concatenated node feature vectors for all given graphs (in one hot encoding).
    """
    # Length of the longest node feature vector
    max_number_of_nodes = max([g.number_of_nodes() for g in graphs])
    # List of all node labels
    node_labels = get_all_node_labels(graphs)

    # Create one hot encoding of graphs
    embeddings = []
    for graph in graphs:
        one_hot_graph = []
        for node in graph.nodes(data=True):
            one_hot_vector = [0] * len(node_labels)
            one_hot_vector[node_labels.index(node[1]["node_label"])] = 1
            one_hot_graph.append(one_hot_vector)
        # Determine how much padding to add
        for _ in range(max_number_of_nodes - len(graph)):
            one_hot_graph.append([0] * len(node_labels))
        embeddings.append(one_hot_graph)

    # Add node attributes to one hot encoding of node label
    if with_attributes:
        # assume node attribute vectors are of the same length
        len_node_attribute = 0
        for i in range(len(graphs)):
            for j, (node_label, node_attributes) in enumerate(graphs[i].nodes(data="node_attributes")):
                # extend node vector by node attributes
                # normalize node_attributes according to l2 norm (default)
                l2_norm = np.linalg.norm(node_attributes)
                node_attributes = node_attributes / l2_norm
                embeddings[i][j].extend(node_attributes)
                len_node_attribute = len(node_attributes)
            # extend padding accordingly
            for j in range(len(graphs[i]), max_number_of_nodes - len(graphs[i])):
                embeddings[i][j].extend([0] * len_node_attribute)

        # final check for padding in embeddings
        for i in range(len(graphs)):
            for j in range(max_number_of_nodes):
                while len(embeddings[i][j]) < (len(node_labels) + len_node_attribute):
                    embeddings[i][j].extend([0])

    # Cast to torch tensor
    embeddings = torch.Tensor(embeddings)
    return embeddings


def get_all_node_labels(graphs: list[nx.Graph]) -> list:
    """
    Extracts all nodes labels from a list of graphs.

    Parameters
    ----------
    graphs List of networkx graphs

    Returns
    -------
    List of unique node labels occurring in the graphs.
    """
    label_set = set()
    for graph in graphs:
        for node in graph.nodes(data=True):
            label_set.add(node[1]["node_label"])
    return list(label_set)


def extract_graph_labels_from_dataset(dataset: Iterable, label_key: str = "label") -> list:
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


def extract_node_labels_from_dataset(dataset: Iterable, label_key: str = "node_label") -> list:
    """
    Gets the node labels of a dataset for datasets that are subscript-able.

    Parameters
    ----------
    dataset The dataset of which to get the labels.
    label_key The key to look at to get the label.

    Returns
    -------
    The list of labels extracted from the dataset.
    """
    return [node[1] for element in dataset for node in element.nodes(data=label_key)]
