import random
import copy

import networkx as nx
import numpy as np

from typing import List
from .graphlets import get_all_graphlets

"""
G: networkx graph
seed: seed for random sampling
return: feature vector with shape (1, 34) as numpy array
"""


def graphlet_kernel(G, seed=12345):
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
    graphs: List[nx.Graph] = list(copy.deepcopy(g))
    feature_vectors = []
    for i in range(len(graphs)):
        feature_vectors.append(graphlet_kernel(graphs[i]).transpose())

        if i == int(0.25 * len(graphs)):
            print("25%")
        elif i == int(0.5 * len(graphs)):
            print("50%")
        elif i == int(0.75 * len(graphs)):
            print("75%")
        elif i == int(len(graphs) - 1):
            print("100%")

    return feature_vectors
