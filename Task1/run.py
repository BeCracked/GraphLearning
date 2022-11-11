import scipy.sparse

from helper.matrix_io import from_adj_str
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
from Kernels.wl_kernel import wl_kernel, print_feature_vectors


def run_kernel():
    parser = ap.ArgumentParser(prog="Graph Kernels",
                               description="Allows to get feature vectors for graphs with different kernels")
    parser.add_argument("kernel", choices=["closed_walk", "graphlet", "WL"])
    parser.add_argument("graphs", nargs="+")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    # Parse graphs
    graphs = []
    for g_str in args.graphs:
        graphs.append(from_adj_str(g_str))

    feature_vectors = []
    match args.kernel:
        case "closed_walk":
            if not args.quiet:
                print(f"Executing WL kernel on {len(graphs)} graphs...")
            return

        case "graphlet":
            if not args.quiet:
                print(f"Executing WL kernel on {len(graphs)} graphs...")
            return

        case "WL":
            if not args.quiet:
                print(f"Executing WL kernel on {len(graphs)} graphs...")
            feature_vectors = wl_kernel(4, graphs)
            if not args.quiet:
                print(f"{args.kernel} gave the following feature vectors:")
            print_feature_vectors(feature_vectors)

        case _:
            print(f"Kernel must be in {['closed_walk', 'graphlet', 'wl']}")
            return


if __name__ == '__main__':
    run_kernel()

'''    s1 = """
    [
    [0,1,1,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [0,0,0,0,0]
    ]
    """
    g1 = from_adj_str(s1)

    f_vec = wl_kernel(4, g1, plot_steps=True)
    print(f_vec)
'''
