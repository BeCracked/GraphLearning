import os
from typing import Optional

import torch
import pickle

from torch.utils.data import DataLoader

import DataHandling.preprocessing as preprocessing
from DataHandling.sparse_graph_dataset import SparseGraphDataset, sparse_graph_collation
from Modules.RNetwork import RNetwork

from tqdm import tqdm

QUIET = os.getenv("QUIET", default=True)


def run_graph_regression(train_data_path: str, test_data_path: str, validation_data_path: str, *,
                         device: Optional[str] = None, epoch_count=10, learning_rate=1e-30,
                         **config) -> tuple[float, float, float]:
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
    validation_loader = get_data_loader(validation_data_path, **config)

    # TODO: Extract node/edge feature dimension

    # Create model
    model = RNetwork(**config)

    model.train()
    model.to(device)

    # Construct optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.functional.l1_loss

    # Train model
    train_mae = val_mae = 0
    best_val_mae = 0
    best_model = None
    for epoch in tqdm(range(epoch_count)):
        model.train()
        train_mae = train_loop(train_loader, model, loss_fn, opt)  # Consider only accuracy of last epoch
        model.eval()
        val_mae = validation(validation_loader, model, loss_fn)
        if best_val_mae == 0:
            best_val_mae = val_mae
            best_model = model.state_dict()
        elif val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model.state_dict()

    model.eval()
    test_mae = validation(test_loader, model, loss_fn)

    return train_mae, best_val_mae, test_mae


def validation(dataloader, model, loss_fn):
    absolute_error = 0
    for batch, (batch_idx, idx_E, H, x_E, y_train) in enumerate(dataloader):
        y_pred = model(H, x_E, idx_E, batch_idx)
        # to remove unnecessary dimension
        y_pred = torch.squeeze(y_pred)
        loss = loss_fn(y_pred, y_train)
        absolute_error += loss

    return abs(absolute_error / len(dataloader))


def get_data_loader(path: str, *, batch_size: int = 128, **config) -> DataLoader:
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
        data = preprocessing.node_labels_to_one_hot(data)

    dataset = SparseGraphDataset(data, **config)
    dataloader = DataLoader(dataset, collate_fn=sparse_graph_collation, batch_size=batch_size)

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
    absolute_error = 0
    for batch, (batch_idx, idx_E, H, x_E, y_train) in enumerate(dataloader):
        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass and loss
        y_pred = model(H, x_E, idx_E, batch_idx)
        # to remove unnecessary dimension
        y_pred = torch.squeeze(y_pred)
        loss = loss_fn(y_pred, y_train)
        absolute_error += loss
        # Backward pass and sgd step
        loss.backward()
        optimizer.step()

    return abs(absolute_error / len(dataloader))


if __name__ == '__main__':
    from Task3.configurations import zinc_base_params

    train_mae, best_val_mae, test_mae = run_graph_regression("datasets/ZINC_Train/data.pkl", "datasets/ZINC_Test/data.pkl", "datasets/ZINC_Val/data.pkl",
                         device="cpu", **zinc_base_params)
    print("train:", train_mae)
    print("val:", best_val_mae)
    print("test:", test_mae)


