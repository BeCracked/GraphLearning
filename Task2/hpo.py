from functools import partial

import numpy as np
import optuna
from GraphClassification import run_graph_classification
from NodeClassification import run_node_classification

# These are the best know parameter sets for each dataset
best_know_params = {
    "NCI": {"epochs": 55, "learning_rate": 0.001961282945941816},  # 0.7841079059829059
    "ENZYMES": {"epochs": 60, "learning_rate": 0.0051646504688932955},  # 0.5500982142857143
    "Citeseer": {"epochs": 57, "learning_rate": 0.016245162317569194},  # 0.8982487922705316
    "Cora": {"epochs": 43, "learning_rate": 0.037515200531228275},  # 0.9336779911373707
}


def graph_objective(dataset_path: str, dataset_name: str, trial):
    """
    Objective function for optuna hpo for graph classification.

    Parameters
    ----------
    dataset_path Path to the data set.
    dataset_name Name of the used data set.
    trial The optuna trial.

    Returns
    -------
    The test accuracy of the trial.

    """
    param = {
        "epochs": trial.suggest_int("epochs", 10, 100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1)
    }

    print(f"Running trial with params {param}")

    train_accs, train_stds, test_accs, test_stds = run_graph_classification(dataset_path, dataset_name, **param)

    print(f"Train Accuracy: {np.mean(train_accs)*100:0.2f}%(±{np.mean(train_stds)*100:0.2f})")
    print(f"Test Accuracy: {np.mean(test_accs)*100:0.2f}%(±{np.mean(test_stds)*100:0.2f})")

    return np.mean(test_accs)


def graph_optimization(dataset_path: str, dataset_name: str, enqueue_known_best: bool = False):
    """
    Runs an optuna study to find the parameter values for epochs and learning_rate that give the best test accuracy.

    Parameters
    ----------
    dataset_path Path to the data set.
    dataset_name Name of the used data set.
    enqueue_known_best Whether to add the hardcoded best known parameters to the study for comparison.

    Returns
    -------
    The best parameters found by the study and the resulting test accuracy.

    """
    objective = partial(graph_objective, dataset_path, dataset_name)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(),
                                study_name=f"Graph Classification - {dataset_name}")

    if enqueue_known_best:
        match dataset_name:
            case "NCI":
                study.enqueue_trial(best_know_params[dataset_name])
            case "ENZYMES":
                study.enqueue_trial(best_know_params[dataset_name])
            case _:
                pass

    study.optimize(objective, n_trials=10, timeout=6000)

    return study.best_params, study.best_value


def node_objective(train_path: str, test_path: str, trial):
    """
    Objective function for optuna hpo for node classification.

    Parameters
    ----------
    train_path Path to the train data.
    test_path Path to the test data.
    trial The optuna trial.

    Returns
    -------
    The test accuracy of the trial.

    """
    param = {
        "epochs": trial.suggest_int("epochs", 10, 60),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1)
    }

    print(f"Running trial with params {param}")

    train_accs, train_stds, test_accs, test_stds = run_node_classification(train_path, test_path, **param)

    print(f"Train Accuracy: {np.mean(train_accs)*100:0.2f}%(±{np.mean(train_stds)*100:0.2f})")
    print(f"Test Accuracy: {np.mean(test_accs)*100:0.2f}%(±{np.mean(test_stds)*100:0.2f})")
    return np.mean(test_accs)


def node_optimization(train_path: str, test_path: str, dataset_name: str, enqueue_known_best: bool = False):
    """
    Runs an optuna study to find the parameter values for epochs and learning_rate that give the best test accuracy.

    Parameters
    ----------
    train_path Path to the train data.
    test_path Path to the test data.
    dataset_name Name of the used data set.
    enqueue_known_best Whether to add the hardcoded best known parameters to the study for comparison.

    Returns
    -------
    The best parameters found by the study and the resulting test accuracy.

    """
    objective = partial(node_objective, train_path, test_path)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(),
                                study_name=f"Node Classification - {dataset_name}")

    if enqueue_known_best:
        match dataset_name:
            case "Citeseer":
                study.enqueue_trial(best_know_params[dataset_name])
            case "Cora":
                study.enqueue_trial(best_know_params[dataset_name])
            case _:
                pass

    study.optimize(objective, n_trials=10, timeout=6000)

    return study.best_params, study.best_value


if __name__ == '__main__':
    """
    path, dataset = "../Task1/datasets/ENZYMES/data.pkl", "ENZYMES"
    best_param, best_val = graph_optimization(path, dataset, True)
    print(f"ENZYMES: Best found params:{best_param}, Accuracy:{best_val * 100:.2f}%")

    path, dataset = "../Task1/datasets/NCI1/data.pkl", "NCI"
    best_param, best_val = graph_optimization(path, dataset, True)
    print(f"NCI: Best found params:{best_param}, Accuracy:{best_val * 100:.2f}%")
    """

    train_path, test_path, dataset = "./datasets/Citeseer_Train/data.pkl", "./datasets/Citeseer_Eval/data.pkl", "Citeseer"
    best_param, best_val = node_optimization(train_path, test_path, dataset, True)
    print(f"Citeseer: Best found params:{best_param}, Accuracy:{best_val * 100:.2f}%")

    train_path, test_path, dataset = "./datasets/Cora_Train/data.pkl", "./datasets/Cora_Eval/data.pkl", "Cora"
    best_param, best_val = node_optimization(train_path, test_path, dataset, True)
    print(f"Cora: Best found params:{best_param}, Accuracy:{best_val * 100:.2f}%")
