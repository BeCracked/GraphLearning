import argparse as ap

from Kernels.wl_kernel import wl_kernel, print_feature_vectors
from Kernels.closed_walk_kernel import closed_walk_kernel
from Kernels.graphlet_kernel import graphlet_kernel
from helper.matrix_io import from_adj_str


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

    match args.kernel:
        case "closed_walk":
            if not args.quiet:
                print(f"Executing closed walk kernel on {len(graphs)} graphs...")
            feature_vectors = []
            for graph in graphs:
                # TODO: need to decide l
                feature_vectors.append(closed_walk_kernel(graph, 10))
            if not args.quiet:
                print(f"{args.kernel} gave the following feature vectors:")
            for vector in feature_vectors:
                print(vector)
            return

        case "graphlet":
            if not args.quiet:
                print(f"Executing graphlets kernel on {len(graphs)} graphs...")
            feature_vectors = []
            for graph in graphs:
                feature_vectors.append(graphlet_kernel(graph))
            if not args.quiet:
                print(f"{args.kernel} gave the following feature vectors:")
            for vector in feature_vectors:
                print(vector)
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
