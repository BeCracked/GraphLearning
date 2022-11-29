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


def graph_optimization(dataset_path: str, dataset_name: str, enqueue_known_best: bool = False):
    objective = partial(graph_objective, dataset_path, dataset_name)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    if enqueue_known_best:
        match dataset_name:
            case "NCI":
                study.enqueue_trial({"epochs": 60, "learning_rate": 0.00641722976851177})  # ~76%
            case "ENZYMES":
                study.enqueue_trial({'epochs': 55, 'learning_rate': 0.0015777811296388503})  # 52.61%
            case _:
                pass

    study.optimize(objective, n_trials=20, timeout=600)

    return study.best_params, study.best_value


if __name__ == '__main__':
    # NCI: {'epochs': 60, 'learning_rate': 0.00641722976851177}
    #path, dataset = "../Task1/datasets/ENZYMES/data.pkl", "ENZYMES"
    path, dataset = "../Task1/datasets/NCI1/data.pkl", "NCI"
    best_param, best_val = graph_optimization(path, dataset, True)
    print(f"Best found params:{best_param}, Accuracy:{best_val * 100:.2f}%")
