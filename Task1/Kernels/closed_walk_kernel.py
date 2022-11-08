import numpy as np
import networkx as nx

"""
information on how to compute number of closed walks of fixed length:
number of closed walks of length i in G equals trace of adjacency matrix 
that is raised by the power of i (according to literature)
"""

"""
G: networkx graph
l: maximum length of closed walks
return: feature vector with shape (1, l+1) as numpy array
"""
def closed_walk_kernel(G, l):
    A = nx.to_numpy_matrix(G)
    # A_exp holds adjacency matrix exponentiated with i (start with exponent 0 which is identity matrix)
    A_exp = np.identity(len(G.nodes))

    # compute feature vector as histogram of closed walks up to length l (see information above)
    feature_vector = np.ndarray((1, l+1))
    for i in range(l+1):
        feature_vector[0, i] = np.trace(A_exp)
        # @ is operator for matrix multiplication
        A_exp = A_exp @ A

    return feature_vector
