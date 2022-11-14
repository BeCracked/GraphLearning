import copy
from hashlib import sha256
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
from multiset import Multiset
from scipy.sparse import csr_matrix


class InjectiveHash:
    """
    Defines a definitely injective hash function which simply counts up an id for unseen values and tracks them.
    """

    def __init__(self):
        self.hash_table = {}
        self.next_id = 0

    def get_hash(self, s: str) -> int:
        if s not in self.hash_table:
            self.hash_table[s] = self.next_id
            self.next_id += 1
        return self.hash_table[s]


def wl_kernel(k: int, g: List[nx.Graph], *, plot_steps=False) -> list[csr_matrix]:
    graphs: List[nx.Graph] = copy.deepcopy(g)  # Copy all graphs as we will modify them during the steps
    feature_vectors = {}
    # Init histogram vectors for each graph, accessible with ids
    for gi in range(len(graphs)):
        # Since we don't know how long our vector will be, we append to lists because it's faster
        feature_vectors[gi] = []

    hash_func = InjectiveHash()

    # Initialise node colors
    unique_colors = set()
    for gi in range(len(graphs)):
        graph = graphs[gi]
        color_mapping = {}
        for node in graph.nodes:
            label = ""
            if isinstance(node, str):
                label = node
            color_id = hash_func.get_hash(label)
            unique_colors.add(color_id)
            color_mapping[node] = {"color_id": color_id, "color": _hash_to_color(str(color_id))}
        nx.set_node_attributes(graph, color_mapping)

        if plot_steps:
            show_colored_graph(graph)

    # Set feature vector for initial colouring
    for gi in range(len(graphs)):
        graph = graphs[gi]
        color_counts = {color: 0 for color in unique_colors}
        for node in graph.nodes("color_id"):
            color_counts[node[1]] += 1
        feature_vectors[gi].extend([color_counts[color] for color in list(sorted(color_counts.keys()))])

    # Execute steps
    for i in range(0, k):
        # Append histogram vectors for current step
        for gi in range(len(graphs)):
            step_vector = _perform_coloring_step(graphs, hash_func, plot_step=plot_steps)
            feature_vectors[gi].extend(step_vector[gi])

    # Fill vectors to equal length
    n = max([len(v) for v in feature_vectors.values()])  # Longest vector
    for gi, feature_vector in feature_vectors.items():
        feature_vector += [0] * (n - len(feature_vector))

    # Transform feature vectors to sparse
    spare_feature_vectors = []
    for i in range(len(feature_vectors)):
        list_vector = feature_vectors[i]
        vector = csr_matrix(list_vector, shape=(1, len(list_vector))).transpose()
        spare_feature_vectors.append(vector)

    return spare_feature_vectors


def _perform_coloring_step(graphs: List[nx.Graph], hash_func: InjectiveHash, *, plot_step: bool = False) \
        -> Dict[int, List[int]]:
    """

    Returns
    -------
    A mapping GraphId -> Feature vector for the current step
    """
    unique_colors = set()
    # Do update step
    for graph in graphs:
        color_mapping = {}
        for node, init_color_id in graph.nodes("color_id"):
            # Construct string representation of node colour and neighborhood
            nb_colors = Multiset([graph.nodes[neighbor]["color_id"] for neighbor in graph.neighbors(node)])
            hash_string = f"{init_color_id}-N:{nb_colors}"

            # Get color hash
            color_id = hash_func.get_hash(hash_string)
            unique_colors.add(color_id)
            color_mapping[node] = {"color_id": color_id, "color": _hash_to_color(str(color_id))}
        # Update information on graphs
        nx.set_node_attributes(graph, color_mapping)

    # Extract feature vector
    feature_vectors = {}
    for gi in range(len(graphs)):
        graph = graphs[gi]
        # Count color ids
        color_counts = {color: 0 for color in unique_colors}
        for node in graph.nodes("color_id"):
            color_counts[node[1]] += 1
        feature_vectors[gi] = [color_counts[color] for color in list(sorted(color_counts.keys()))]

    if plot_step:
        for graph in graphs:
            show_colored_graph(graph)

    return feature_vectors


def _hash_to_color(s: str) -> Tuple[int, int, int]:
    """
    Hash a string into a color. Does _not_ perform collision correction.
    Parameters
    ----------
    s           - The string to hash into a color
    color_seed - Random seed for deterministic but varied colors

    Returns
    -------
    An RGB tuple representing a color.
    """
    # Scramble input using sha256 for uniform distribution
    hash_bytes = sha256(s.encode("UTF-8")).digest()
    count = len(hash_bytes)
    r = g = b = 0
    for i in range(0, count, 3):
        # Add bytes round-robin style onto r, g and b modulo 256
        r = (r + hash_bytes[i]) % 256
        # To avoid out of bounds exceptions we are simply content with double sampling the first two values
        g = (g + hash_bytes[(i + 1) % count]) % 256
        b = (b + hash_bytes[(i + 2) % count]) % 256

    return r, g, b


def t_hash():
    from collections import defaultdict
    alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z']
    hash_counts = defaultdict(lambda: 0)
    for l1 in alpha:
        c = _hash_to_color(l1)
        hash_counts[c] += 1
        for l2 in alpha:
            c = _hash_to_color(l1 + l2)
            hash_counts[c] += 1
            for l3 in alpha:
                c = _hash_to_color(l1 + l2 + l3)
                hash_counts[c] += 1
    print("Collision count: ", len({col: count for col, count in hash_counts.items() if count > 1}))
    print({col: count for col, count in hash_counts.items() if count > 1})


def print_feature_vectors(feature_vectors: List[csr_matrix]):
    # c_id G0  G1
    # 0     5   5
    # 1     3   2
    # 2     1   1
    # 3     1   1
    # 4     2   1

    vec_range = range(0, feature_vectors[0].get_shape()[0])
    print(f"c_id ", end="")
    for i in range(len(feature_vectors)):
        print(f"G{i: <3}", end="")
    print()
    for j in vec_range:
        print(f"{j: <3}", end="")
        for vector in feature_vectors:
            print(f"{vector.getrow(j).toarray()[0][0]: 4}", end="")
        print()


def show_colored_graph(graph: nx.Graph) -> None:
    node_colors = [tuple(ti / 256 for ti in color) for node, color in graph.nodes("color")]
    label_mapping = {node: color_id for node, color_id in graph.nodes("color_id")}
    nx.draw_networkx(graph, pos=nx.planar_layout(graph), node_size=800, node_color=node_colors, labels=label_mapping)
    plt.show()


if __name__ == '__main__':
    from Task1.helper.graph_gen import get_random_graph

    G1 = get_random_graph(123147)
    G2 = get_random_graph(123148)

    G5 = get_random_graph(123146)

    vectors = wl_kernel(4, [G1, G2, G5], plot_steps=False)

    for r in range(len(vectors)):
        print(f"G{r + 1} feature vector:")
        #print(vectors[r])
        print(vectors[r].get_shape())
