import os
from typing import Optional

import numpy as np
import torch
import pickle

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import DataHandling.preprocessing as preprocessing
from DataHandling.sparse_graph_dataset import SparseGraphDataset, sparse_graph_collation

QUIET = os.getenv("QUIET", default=True)


def run_graph_regression(train_data_path: str, test_data_path: str, *,
                             device: Optional[str] = None,
                             epochs=10, learning_rate=1e-30, **config) -> tuple[float, float]:
    """
    Performs k-fold cross validation with the graph classification net on the given dataset.

    Parameters
    ----------
    path The path to the dataset file.
    dataset_name Name of the dataset to use. Either "NCI" or "ENZYMES". Determines certain import behaviour.
    device The device name to run on.
    epochs Number of epochs to train for.
    batch_size Batch sizes for the DataLoaders to use.
    learning_rate Learning rate to use by the optimizer.
    config Configuration dictionary that may be used in other modules or functions.

    Returns
    -------
    The train and test accuracy
    Tuple of Lists of the form (train_accs, train_stds, test_accs, test_stds).
    """
    torch.manual_seed(42)
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader = get_data_loader(train_data_path, **config)
    test_loader = get_data_loader(test_data_path, **config)
    # todo: load validation set

    # Create model
    model: torch.Module = None  # TODO: Create model class from torch.Module (remember to use **config)
    # Todo: frage, An dieser Stelle das graph regression model erzeugen?
    # In alter Abgabe:  model = GraphLevelGCN(input_dim=len(x[0][0]), output_dim=num_labels, hidden_dim=64)
    model.train()
    model.to(device)

    # Construct optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.functional.cross_entropy

    # Train model
    train_acc, train_std = 0, 0
    for epoch in tqdm(range(epochs), disable=QUIET):
        train_acc, train_std = train_loop(train_loader, model, loss_fn, opt)  # Consider only accuracy of last epoch
        # todo: use validation set to compute MAE and save it to report it in the readme file

    test_acc, test_std = test_loop(test_loader, model)

    return train_acc, test_acc


def get_data_loader(path: str, **config) -> DataLoader:
    """
    Load graphs of dataset, apply normalization to adjacency matrices and extract labels.
    ----------
    path The path to the dataset
    config Configuration dictionary that may be used in other modules or functions.

    Returns
    -------
    Node Feature Embeddings and normalized matrices as Tensors.
    Labels as Tensors and number of labels.
    """
    with open(path, 'rb') as f:
        data = preprocessing.edge_labels_to_one_hot(pickle.load(f))

    dataset = SparseGraphDataset(data, **config)
    dataloader = DataLoader(dataset, collate_fn=sparse_graph_collation)

    return dataloader


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader The dataloader to use. Needs to provide x: node embeddings, a: normalized adjacency matrices, y: labels.
    model The model to train.
    loss_fn The loss function to use. E.g., torch.nn.functional.cross_entropy.
    optimizer The optimizer to use. E.g., torch.optim.Adam(model.parameters(), lr=learning_rate).

    Returns
    -------
    The list of accuracy values for each batch.
    """
    accuracies = np.zeros(len(dataloader))
    for batch, (idx_E, x_V, x_E, y_train) in enumerate(dataloader):
        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass and loss
        y_pred = model(idx_E, x_V, x_E)
        loss = loss_fn(y_pred, y_train)

        # Backward pass and sgd step
        loss.backward()
        optimizer.step()

        # Record accuracy
        accuracies[batch] = get_accuracy(y_pred, y_train)

    return accuracies.mean(), accuracies.std()


def test_loop(dataloader, model):
    accuracies = np.zeros(len(dataloader))
    with torch.no_grad():
        for batch, (x_test, a_test, y_test) in enumerate(dataloader):
            y_pred = model(x_test, a_test)
            accuracies[batch] = get_accuracy(y_pred, y_test)

    return accuracies.mean(), accuracies.std()


def get_accuracy(pred: Tensor, truth: Tensor) -> float:
    correct = (pred.argmax(1) == truth).type(torch.float).sum().item()
    return correct / len(truth)


if __name__ == '__main__':
    from Task3.configurations import zinc_base_params
    run_graph_classification("datasets/ZINC_Train/data.pkl", "datasets/ZINC_Test/data.pkl", **zinc_base_params)
