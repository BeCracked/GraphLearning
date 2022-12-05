import pickle
import networkx as nx
from typing import List


def transform_edge_labels(graphs: List[nx.Graph]):
    edge_labels = get_edge_labels(graphs)
    new_graphs = []
    for graph in graphs:
        new_labels = dict()
        for edge in graph.edges(data=True):
            new_labels[(edge[0], edge[1])] = edge[2]["edge_label"]

        for (v, w), value in new_labels.items():
            one_hot = [0] * len(edge_labels)
            one_hot[edge_labels.index(new_labels[(v, w)])] = 1
            new_labels[(v, w)] = one_hot

        new_graph = graph
        nx.set_edge_attributes(new_graph, new_labels, name="edge_label")
        new_graphs.append(new_graph)

    return new_graphs


def get_edge_labels(graphs: List[nx.Graph]):
    labels = set()
    for graph in graphs:
        for edge in graph.edges(data=True):
            labels.add(edge[2]["edge_label"])

    return list(labels)


if __name__ == '__main__':
    with open("../datasets/ZINC_Train/data.pkl", "rb") as f:
        data = pickle.load(f)

    transform_edge_labels(data)