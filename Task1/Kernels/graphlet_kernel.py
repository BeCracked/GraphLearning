import random

import networkx as nx
import numpy as np

import graphlets

"""
G: networkx graph
seed: seed for random sampling
return: feature vector with shape (1, 34) as numpy array
"""


def graphlet_kernel(G, seed=12345):
    random.seed(seed)
    graphlets_list = graphlets.get_all_graphlets()

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
