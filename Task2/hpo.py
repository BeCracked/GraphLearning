from functools import partial

import numpy as np
import optuna
from GraphClassification import run_graph_classification
from NodeClassification import run_node_classification


def graph_objective(dataset_path: str, dataset_name: str, trial):
    param = {
        "epochs": trial.suggest_int("epochs", 10, 60),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1)
    }

    print(f"Running trial with params {param}")

    train_accs, train_stds, test_accs, test_stds = run_graph_classification(dataset_path, dataset_name, **param)

    print(f"Train Accuracy: {np.mean(train_accs)*100:0.2f}%(±{np.mean(train_stds)*100:0.2f})")
    print(f"Test Accuracy: {np.mean(test_accs)*100:0.2f}%(±{np.mean(test_stds)*100:0.2f})")

    return np.mean(test_accs)


def graph_optimization(dataset_path: str, dataset_name: str, enqueue_known_best: bool = False):
    objective = partial(graph_objective, dataset_path, dataset_name)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    if enqueue_known_best:
        match dataset_name:
            case "NCI":
                study.enqueue_trial({"epochs": 55, "learning_rate": 0.001961282945941816})  # 0.7757567663817664
            case "ENZYMES":
                study.enqueue_trial({'epochs': 60, 'learning_rate': 0.0051646504688932955})  # 0.5510982142857143
            case _:
                pass

    study.optimize(objective, n_trials=20, timeout=600)

    return study.best_params, study.best_value


def node_objective(dataset_path: str, dataset_name: str, trial):
    param = {
        "epochs": trial.suggest_int("epochs", 10, 60),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1)
    }

    print(f"Running trial with params {param}")

    train_accs, train_stds, test_accs, test_stds = run_node_classification(dataset_path, dataset_name, **param)

    print(f"Train Accuracy: {np.mean(train_accs)*100:0.2f}%(±{np.mean(train_stds)*100:0.2f})")
    print(f"Test Accuracy: {np.mean(test_accs)*100:0.2f}%(±{np.mean(test_stds)*100:0.2f})")
    return np.mean(test_accs)


def node_optimization(train_path: str, test_path: str, dataset_name: str, enqueue_known_best: bool = False):
    objective = partial(node_objective, train_path, test_path)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    if enqueue_known_best:
        match dataset_name:
            case "Citeseer":
                study.enqueue_trial({"epochs": 57, "learning_rate": 0.8982487922705316})  # 0.8982487922705316
                pass
            case "Cora":
                #study.enqueue_trial({'epochs': 60, 'learning_rate': 0.0051646504688932955})  # 0.5510982142857143
                pass
            case _:
                pass

    study.optimize(objective, n_trials=30, timeout=600)

    return study.best_params, study.best_value


if __name__ == '__main__':
    path, dataset = "../Task1/datasets/ENZYMES/data.pkl", "ENZYMES"
    # path, dataset = "../Task1/datasets/NCI1/data.pkl", "NCI"
    best_param, best_val = graph_optimization(path, dataset, True)
    #train_path, test_path, dataset = "./datasets/Citeseer_Train/data.pkl", "./datasets/Citeseer_Eval/data.pkl", "Citeseer"
    #best_param, best_val = node_optimization(train_path, test_path, dataset, True)
    print(f"Best found params:{best_param}, Accuracy:{best_val * 100:.2f}%")
