# Task 2

## Repository Structure

The python environment is described in the `requirements.txt` and can be installed with the following command:
```pip install -r requirements.txt```

Files relevant to `Task2`:

* `Modules` folder: Contains the PyTorch modules which form the layers and nets of the Graph Neural Networks for Graph and Node Classification.
* `datasets` folder: Contains the datasets new to Task 2 (Citeseer and Core ). Other datasets can be found in `Task1/datasets`.
* `preprocessing.py` file: Contains methods for extracting data labels and the calculation of the normalized adjacency matrices.
* `hpo.py` file: Contains the logic for performing hyperparameter optimization with optuna.
* `run.py` file: Contains the argument parsing logic for the CLI.

## How To Run Scripts


## Evaluation Results

### Graph Classification
|         | Train Accuracy | Test Accuracy  |
|---------|----------------|----------------|
| ENZYMES | 66.61% (±8.07) | 55.01% (±5.11) |
| NCI1    | 84.25% (±6.65) | 78.41% (±6.18) |

### Node Classification
|          | Train Accuracy | Test Accuracy  |
|----------|----------------|----------------|
| Citeseer | 94.51% (±0.00) | 89.80% (±0.00) |
| Cora     | 96.79% (±0.00) | 93.37% (±0.00) |

### Hyper Parameter
Hyper Parameter Optimization was performed for the parameters `epochs` and `learning_rate`.
A dictionary of the best found parameters with which the above results were obtained:
```json
{
    "NCI": {"epochs": 55, "learning_rate": 0.001961282945941816},
    "ENZYMES": {"epochs": 60, "learning_rate": 0.0051646504688932955},
    "Citeseer": {"epochs": 57, "learning_rate": 0.016245162317569194},
    "Cora": {"epochs": 43, "learning_rate": 0.037515200531228275}
}
```

