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
                         **config) -> tuple[float, float, float, float]:
    """
    Performs graph regression on the given training, validation and test dataset.

    Parameters
    ----------
    train_data_path The path to the training dataset file.
    test_data_path The path to the test dataset file.
    validation_data_path The path to the validation dataset file.
    device The device name to run on.
    epoch_count Number of epochs to train for.
    learning_rate Learning rate to use by the optimizer.
    config Configuration dictionary that may be used in other modules or functions.

    Returns
    -------
    The train, validation and test mean absolute error.
    """
    torch.manual_seed(42)
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader = get_data_loader(train_data_path, **config)
    test_loader = get_data_loader(test_data_path, **config)
    validation_loader = get_data_loader(validation_data_path, **config)

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
    path = os.path.abspath(os.getcwd()) + "/best_model"

    foo = False
    best_number_of_epochs = 0
    for epoch in tqdm(range(epoch_count)):
        model.train()
        train_mae = train_loop(train_loader, model, loss_fn, opt)  # Consider only accuracy of last epoch
        model.eval()
        val_mae = validation(validation_loader, model, loss_fn)
        if best_val_mae == 0:
            best_val_mae = val_mae
            torch.save(model.state_dict(), path)
        elif val_mae < best_val_mae:
            print(val_mae)
            path2 = os.path.abspath(os.getcwd()) + "/best_model"
            best_val_mae = val_mae
            torch.save(model.state_dict(), path2)
            foo = True
            best_number_of_epochs = epoch
    if foo:
        selected_model = RNetwork(**config)
        path3 = os.path.abspath(os.getcwd()) + "/best_model"
        selected_model.load_state_dict(torch.load(path3))
    else:
        selected_model = model

    selected_model.eval()
    test_mae = validation(test_loader, selected_model, loss_fn)
    val_mae = validation(validation_loader, selected_model, loss_fn)
    train_mae = train_loop(train_loader, selected_model, loss_fn, opt)

    model.eval()
    print(" test mae: ", test_mae)
    return train_mae, val_mae, test_mae, best_number_of_epochs


def validation(dataloader, model, loss_fn):
    """
    Validates/Tests the trained model.

    Parameters
    ----------
    dataloader Data of validation or test dataset.
    model Trained model to be validated/tested.
    loss_fn Calculates the absolute error (l1-loss).

    Returns
    -------
    The validation/test mean absolute error.
    """
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
    Loads graphs of dataset, applies one hot transformation of node and edge labels.
    ----------
    path The path to the dataset.
    batch_size Size of the batch.
    config Configuration dictionary that may be used in other modules or functions.

    Returns
    -------
    Dataloader with the prepared dataset for the graph regression.
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
    dataloader The dataloader to use.
    model The model to train.
    loss_fn The loss function to use (l1-loss in our case).
    optimizer The optimizer to use. (Adam in our case).

    Returns
    -------
    The training mean absolute error.
    """
    absolute_error_sum = 0
    for batch, (batch_idx, idx_E, H, x_E, y_train) in enumerate(dataloader):
        # Set gradients to zero
        optimizer.zero_grad()
        # Forward pass and loss
        y_pred = model(H, x_E, idx_E, batch_idx)
        # to remove unnecessary dimension
        y_pred = torch.squeeze(y_pred)

        loss = loss_fn(y_pred, y_train)

        absolute_error_sum += loss
        # Backward pass and sgd step
        loss.backward()
        optimizer.step()

    return abs(absolute_error_sum / len(dataloader))


if __name__ == '__main__':
    from configurations import zinc_base_params

    best_score = 100
    best_params = {}
    print(zinc_base_params)
    train_mae, val_mae, test_mae, e = run_graph_regression("datasets/ZINC_Train/data.pkl",
                                                                        "datasets/ZINC_Test/data.pkl",
                                                                         "datasets/ZINC_Val/data.pkl",
                                                                         device="cpu", **zinc_base_params)
    print(train_mae, test_mae, test_mae)
    """
    for vnode in [False]:
        for aggregation in ["SUM"]:
            for hidden_dim in [40]:
                for layer_count in [7]:
                    for lear_rate in [1e-3]:
                        for batchsize in [128]:
                            for drop_prob in [0.005]:
                                # print(f"dim {hidden_dim}, layer count {layer_count}, learning rate {lear_rate}")
                                zinc_base_params["hidden_dim"] = hidden_dim
                                zinc_base_params["aggregation"] = aggregation
                                zinc_base_params["drop_prob"] = drop_prob
                                zinc_base_params["virtual_node"] = vnode
                                zinc_base_params["learning_rate"] = lear_rate
                                zinc_base_params["batch_size"] = batchsize
                                train_mae, best_val_mae, val_mae, test_mae, e = run_graph_regression(
                                    "datasets/ZINC_Train/data.pkl",
                                    "datasets/ZINC_Test/data.pkl",
                                    "datasets/ZINC_Val/data.pkl",
                                    device="cpu", **zinc_base_params)

                                if test_mae < best_score:
                                    best_score = test_mae
                                    best_params = zinc_base_params

                                    print("9: best score: ", train_mae, best_val_mae, val_mae, test_mae)
                                    print(f"with params {best_params} and {e} epochs")

        print("best overall score: ", best_score)
        print(f"with params {best_params}")

    """
