import argparse as ap
import numpy as np
from GraphClassification import run_graph_classification
from NodeClassification import run_node_classification
from hpo import graph_optimization, node_optimization, best_know_params


def run():
    """
    Argument parsing for the CLI.

    """
    parser = ap.ArgumentParser(prog="Task 2 - Graph Classification",
                               description="Train GCNs for graph classification.")
    parser.add_argument("classification", choices=["graph", "node"])
    parser.add_argument("dataset_name", choices=["ENZYMES", "NCI", "Citeseer", "Cora"])
    # Hyper Parameter Optimization
    parser.add_argument("--hpo", action="store_true", default=False)

    args = parser.parse_args()

    match args.classification:
        case "graph":
            graph_class(args.dataset_name, hpo=args.hpo)
        case "node":
            node_class(args.dataset_name, hpo=args.hpo)


def graph_class(dataset: str, *, hpo: bool = False):
    dataset_dir = "../Task1/datasets"
    data_path = None
    params = {}
    match dataset:
        case "ENZYMES":
            data_path = f"{dataset_dir}/ENZYMES/data.pkl"
            params = best_know_params[dataset]
        case "NCI":
            data_path = f"{dataset_dir}/NCI1/data.pkl"
            params = best_know_params[dataset]
        case _:
            print(f"Error: {dataset} is not a valid dataset")
            return 1

    if hpo:
        graph_optimization(data_path, dataset)
    else:
        train_accs, train_stds, test_accs, test_stds = run_graph_classification(data_path, dataset, **params)
        print(f"Train Accuracy: {np.mean(train_accs) * 100:0.2f}%(±{np.mean(train_stds) * 100:0.2f})")
        print(f"Test Accuracy: {np.mean(test_accs) * 100:0.2f}%(±{np.mean(test_stds) * 100:0.2f})")


def node_class(dataset: str, *, hpo: bool = False):
    dataset_dir = "./datasets"
    train_path = None
    test_path = None
    params = {}
    match dataset:
        case "Citeseer":
            train_path = f"{dataset_dir}/Citeseer_Train/data.pkl"
            test_path = f"{dataset_dir}/Citeseer_Eval/data.pkl"
            params = best_know_params[dataset]
        case "Cora":
            train_path = f"{dataset_dir}/Cora_Train/data.pkl"
            test_path = f"{dataset_dir}/Cora_Eval/data.pkl"
            params = best_know_params[dataset]
        case _:
            print(f"Error: {dataset} is not a valid dataset")
            return 1

    if hpo:
        node_optimization(train_path, test_path, dataset)
    else:
        train_accs, train_stds, test_accs, test_stds = run_node_classification(train_path, test_path, **params)
        print(f"Train Accuracy: {np.mean(train_accs)*100:0.2f}%(±{np.mean(train_stds)*100:0.2f})")
        print(f"Test Accuracy: {np.mean(test_accs)*100:0.2f}%(±{np.mean(test_stds)*100:0.2f})")
    pass


if __name__ == '__main__':
    run()
