# Task 2

## Repository Structure

The python environment is described in the `requirements.txt` and can be installed with the following command:

```pip install -r requirements.txt```

Everything else that is relevant for the second exercise can be found in folder `Task2`:

* `Modules` folder: contains the PyTorch modules which includes
	* the layer components (`GCNLayer.py` and `NormalLayer.py`) 
	* the network components (`GNetwork.py` and `MLP.py`)
	* the GCNs (`GraphLevelGCN.py` and `NodeLevelGCN.py`)
* `datasets` folder: contains the datasets new to Task 2 (Citeseer and Core ). The other datasets (NCI1 and ENZYMES) can be found in `Task1/datasets`.
* `preprocessing.py` file: contains methods for creating the node feature embeddings and the calculation of the normalized adjacency matrices as well as label extractions.
* `GraphClassification.py` file: contains the training and testing of the graph-level GCN.
* `NodeClassification.py` file: contains the training and testing of the node-level GCN.
* `hpo.py` file: contains the logic for performing hyperparameter optimization with optuna.
* `run.py` file: contains the argument parsing logic for the CLI.

## How To Run Scripts
From within the `Task2` directory execute
```bash
python run.py classification dataset [--hpo]
```

| Parameter       | Possible Values                              | Default | Description                                 |
|-----------------|----------------------------------------------|---------|---------------------------------------------|
| `classificaton` | `"graph"`,`"node"`                           | n/a     | What to classify                            |
| `dataset`       | `"ENZYMES"`, `"NCI"`, `"Citeseer"`, `"Cora"` | n/a     | On what dataset
| `--hpo`         | `True`, `False`                              | `False` | Whether to run Hyper Parameter Optimization |

## Evaluation Results

### Graph Classification
| Dataset | Train Accuracy | Test Accuracy  |
|---------|----------------|----------------|
| ENZYMES | 66.61% (±8.07) | 55.01% (±5.11) |
| NCI1    | 84.25% (±6.65) | 78.41% (±6.18) |

### Node Classification
| Dataset  | Train Accuracy | Test Accuracy  |
|----------|----------------|----------------|
| Citeseer | 94.51% (±0.00) | 89.80% (±0.00) |
| Cora     | 96.79% (±0.00) | 93.37% (±0.00) |

### Hyper Parameters
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