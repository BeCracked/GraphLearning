import random
import copy

import networkx as nx
import numpy as np

from typing import List
from .graphlets import get_all_graphlets


def graphlet_kernel(G, seed=12345):
    """
    G: networkx graph
    seed: seed for random sampling
    return: feature vector with shape (1, 34) as numpy array
    """

    random.seed(seed)
    graphlets_list = get_all_graphlets()

    # compute histogram that holds how often graphlet occurs (repeat 1000 times)
    feature_vector = np.zeros((1, 34))
    for i in range(1000):
        # sample 5 nodes to obtain sample graph
        node_sample = random.sample(G.nodes, 5)
        graph_sample = G.subgraph(node_sample)
        # check to which graphlet the sampled graph is isomorphic to
        for j in range(34):
            if nx.is_isomorphic(graph_sample, graphlets_list[j]):
                feature_vector[0, j] += 1
                break

    return feature_vector


def run_graphlet_kernel(*g: nx.Graph):
    """
    g: list of graphs from datasets
    return: numpy matrix with all feature vectors with shape (num_samples, 34)
    """

    graphs: List[nx.Graph] = list(copy.deepcopy(g))
    feature_vectors = []
    for i in range(len(graphs)):
        feature_vectors.append(graphlet_kernel(graphs[i]))

    return np.concatenate(feature_vectors, axis=0)
