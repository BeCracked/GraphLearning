from functools import partial

import numpy as np
import optuna
from GraphClassification import run_graph_classification


def graph_objective(dataset_path: str, dataset_name: str, trial):
    param = {
        "epochs": trial.suggest_int("epochs", 10, 60, step=5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1)
    }

    print(f"Running trial with params {param}")

    train_accs, train_stds, test_accs, test_stds = run_graph_classification(dataset_path, dataset_name, device="cuda",
                                                                            **param)

    return np.mean(test_accs)


def graph_optimization(dataset_path: str, dataset_name: str):
    objective = partial(graph_objective, dataset_path, dataset_name)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    study.optimize(objective, n_trials=10, timeout=600)
    best_params = study.best_params

    return best_params


if __name__ == '__main__':
    # study.enqueue_trial({'epochs': 55, 'learning_rate': 0.005868098030918391})
    path, dataset = "../Task1/datasets/NCI1/data.pkl", "NCI"
    best = graph_optimization(path, dataset)
    print(best)
