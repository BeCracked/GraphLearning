import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix


def norm_adj_matrix(g: nx.Graph, pad_length: None | int = 0) -> coo_matrix:
    """
    Computes the normalized adjacency matrix for graph g with 1/sqrt(d_i*d_j) if (v_i,v_j) \\in E or i=j.
    Parameters
    ----------
    g: The graph of which to compute the normalized adjacency matrix.
    pad_length: The length to pad the adjacency matrix to. Ignored if len(g.nodes) >= pad_length or pad_length = None.

    Returns
    -------
    The normalized adjacency matrix as a dense numpy array with shape (len(g.nodes), len(g.nodes)).
    """
    # Construct vector of degree+1 of all nodes
    degree_vec = np.array([val+1 for (node, val) in g.degree()], dtype=int, ndmin=2)

    # Incidence matrix with diag set to one
    # e[i][j] = 1 iff (v_i,v_j) in E or i=j
    # Note: Casting go lil for construction (setting the diagonal) and casting back should be faster
    e = nx.adjacency_matrix(g, None, dtype=int).tolil()
    e.setdiag(1)
    e = e.tocsr()
    # Matrix of d_i*d_j
    d_m = np.matmul(degree_vec.transpose(), degree_vec)
    # Normalized adjacency matrix
    na_m: coo_matrix = e.multiply(d_m).sqrt()
    # Take reciprocal by modifying data directly
    na_m.data = np.reciprocal(na_m.data)

    # Add padding
    if pad_length and g.number_of_nodes() < pad_length:
        na_m = na_m.reshape((pad_length, pad_length))
    return na_m


if __name__ == '__main__':
    from Task1.helper.graph_gen import get_random_graph

    for i in range(20):
        G = get_random_graph(12345+i)
        m = norm_adj_matrix(G, 15)
        print(len(G.nodes))
        print(m.toarray())
        print("####################################")
