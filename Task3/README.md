# Task 3

## Repository Structure

The python environment is described in the `requirements.txt` and can be installed with the following command:

```pip install -r requirements.txt```

Everything else that is relevant for the third exercise can be found in folder `Task3`:

* DataHandling: contains the preprocessing of the datasets (`preprocessing.py`) and the custom dataset and collation function (`sparse_graph_dataset.py`).
* `Modules` folder: contains the PyTorch modules which includes
	* the GNN layer (`sparse_gnn_layer.py`) 
	* the sum pooling component (`sparse_sum_pooling.py`)
	* the virtual node component (`virtual_node.py`)
	* the complete network (`RNetwork.py`)
* `datasets` folder: contains the training, validation and test dataset (ZINC).
* `configurations.py` file: contains a parameter dictionary that can be used as is or as a basis for new configurations (optimization).
* `graph_regression.py` file: contains the training, validation and testing of the network.

## How To Run Scripts
From within the `Task3` directory execute:

```python graph_regression.py```

## Evaluation Results

| Dataset | Training MAE   | Validation MAE  | Test MAE    |
|---------|----------------|-----------------|-------------|
| ZINC    |                |                 |             |

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