from typing import Optional

import networkx as nx
import numpy as np
from numpy.random import randint

from sparse_graph_dataset import SparseGraphDataset

import pytest

from Task3.configurations import zinc_base_params

gen_settings = {
    "min_node_count": 5,
    "max_node_count": 20,
}
full_settings = zinc_base_params.copy()
full_settings.update(gen_settings)


@pytest.mark.parametrize("random_seeds", [[12345, 67890, 1029384756]])
def test_sanity_check(random_seeds):
    graph_list = []
    for random_seed in random_seeds:
        graph_list.append(random_graph(random_seed, **full_settings))

    ds = SparseGraphDataset(graph_list, **full_settings)

    for i in range(len(ds)):
        edge_list, node_features, edge_features, graph_labels = ds[i]
        assert edge_list.shape[1] == edge_features.shape[0]
        assert isinstance(graph_labels, int)


def random_graph(seed: int, *,
                 min_node_count: int = 2, max_node_count: int = 10, generate_self_loops: bool = False,
                 graph_feature_key: Optional[str] = "label", node_feature_key: Optional[str] = "node_label",
                 edge_feature_key: Optional[str] = "edge_label",
                 **kwargs) -> nx.Graph:
    """
    Generate a random graph with the given parameters based on the seed.
    Will return the same graph for the same seed. Feature keys will be populated randomly.

    """
    np.random.seed(seed)
    node_count = randint(min_node_count, max_node_count)
    adj = randint(0, 2, (node_count, node_count))  # Generate random square binary matrix
    if not generate_self_loops:
        np.fill_diagonal(adj, 0)
    G: nx.Graph = nx.from_numpy_matrix(adj)
    if graph_feature_key:
        G.graph[graph_feature_key] = randint(0, 10)
    if node_feature_key:
        nx.set_node_attributes(G, {n: randint(0, max_node_count) for n in G.nodes}, name=node_feature_key)
    if edge_feature_key:
        nx.set_edge_attributes(G, {e: random_one_hot(randint(0, max_node_count), max_node_count) for e in G.edges},
                               name=edge_feature_key)

    return G


def random_one_hot(val: int, length: int) -> list[int]:
    """
    Generate a one hot encoding vector for a given integer.

    Parameters
    ----------
    val Values to encode
    length The length of the vector. Upper limit of val

    """
    vector = [0] * length
    vector[val] = 1
    return vector
