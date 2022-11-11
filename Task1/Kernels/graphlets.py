import networkx as nx

"""
note that in this case we only consider graphs with 5 nodes (34 graphlets)
return: list with 34 distinct graphlets
"""


def get_all_graphlets():
    graphlets = []

    # 0 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    graphlets.append(G)

    # 1 edge
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2)])
    graphlets.append(G)

    # 2 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (3, 4)])
    graphlets.append(G)

    # 3 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 4), (1, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 4), (3, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (3, 4), (1, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 3), (1, 4), (3, 4)])
    graphlets.append(G)

    # 4 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (3, 4), (1, 4), (1, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (3, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (4, 5), (1, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 5), (2, 3), (4, 5), (1, 4)])
    graphlets.append(G)

    # 5 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (3, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (1, 4), (1, 5), (3, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 5), (1, 3), (1, 4), (2, 3), (3, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 5), (1, 2), (1, 4), (2, 3), (4, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)])
    graphlets.append(G)

    # 6 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (3, 5), (1, 4), (1, 5), (2, 3), (3, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (4, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1, 4)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 4), (1, 5), (3, 5), (3, 4), (4, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 3), (1, 4), (1, 5), (3, 5), (3, 4), (4, 5)])
    graphlets.append(G)

    # 7 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (3, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (4, 5), (3, 4), (3, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1, 4), (2, 5)])
    graphlets.append(G)

    # 8 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1, 3), (1, 4), (3, 5)])
    graphlets.append(G)

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1, 3), (1, 4), (2, 5)])
    graphlets.append(G)

    # 9 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1, 3), (1, 4), (2, 5), (3, 5)])
    graphlets.append(G)

    # 10 edges
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1, 3), (1, 4), (2, 5), (3, 5), (2, 4)])
    graphlets.append(G)

    return graphlets
