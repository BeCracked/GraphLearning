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
Hyper Parameter Optimization was performed for the parameters `epochs`, `learning_rate`, `hidden dimension`, 
`number of layers`, `virtual nodes`, `batch size`, `aggregation type`.

For each hyperparameter we evaluated values within the following ranges:

| Parameter          | Value Range    | 
|--------------------|----------------|
| Number Of Epochs   | 5 - 400        |
| Learning Rate      | 1e-3 - 1e-40   |
| Hidden Dimension   | 10 - 60        |
| Number Of Layers   | 4 - 10         |
| Virtual Nodes      | True and False |
| Batch Size         | 64, 128, 256   |
| Aggregation Type   | SUM, MEAN, MAX |

We found the results were produced by models with a high number of epoches, a learning rate of about 1e-4, 
a hidden dimension of about 40, about 7 layers, no virtual nodes, a batch size around 128 and with the aggregation type
set to SUM.

Our final, best model has the following parameters:

| Parameter          | Value Range | 
|--------------------|-------------|
| Number Of Epochs   |             |
| Learning Rate      |             |
| Hidden Dimension   |             |
| Number Of Layers   |             |
| Virtual Nodes      | False       |
| Batch Size         | 128         |
| Aggregation Type   | SUM         |

and the following evaluation scores:

| Dataset | Training MAE   | Validation MAE  | Test MAE    |
|---------|----------------|-----------------|-------------|
| ZINC    |                |                 |             |

To load the best model, run the following commands:

```
model = RNetwork(**config)
path = os.path.abspath(os.getcwd()) + "/best_model"
model.load_state_dict(torch.load(path))
```
