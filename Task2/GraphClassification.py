import os

import numpy as np
import torch
import pickle

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import KFold
from tqdm import tqdm

import preprocessing
from Task2.Modules.GraphLevelGCN import GraphLevelGCN

QUIET = os.getenv("QUIET", default=True)


def load_data(path: str, dataset: str):
    """
    Load graphs of dataset, apply normalization to adjacency matrices and extract labels.
    ----------
    path Path to dataset
    dataset Either 'NCI' or 'ENZYMES'

    Returns
    -------
    Normalized matrices as Tensors.
    Labels as Tensors.
    """
    # Load dataset
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Normalize adjacency matrices and cast to tensor
    norm_matrices = preprocessing.norm_adj_matrices(data)
    # Obtain node feature embeddings and cast to tensor
    node_features = torch.zeros(1)
    if dataset == "NCI":
        node_features = preprocessing.get_node_feature_embeddings(data, False)
    if dataset == "ENZYMES":
        node_features = preprocessing.get_node_feature_embeddings(data, True)

    # Extract labels and cast to tensor
    labels = preprocessing.extract_graph_labels_from_dataset(data)
    possible_labels = list(set(labels))
    num_labels = len(possible_labels)
    # cross entropy takes indices of labels, not labels itself
    labels = [possible_labels.index(label) for label in labels]
    labels = torch.tensor(labels)

    return node_features, norm_matrices, labels, num_labels


def run_graph_classification(path: str, dataset_name: str, *,
                             device: str|None = None,
                             epochs=10, batch_size=32, learning_rate=1e-3, fold_count=10) \
        -> tuple[list, list, list, list]:
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
    fold_count Fold count for k-fold cross validation.

    Returns
    -------
    The accuracy results for each fold.
    Tuple of Lists of the form (train_accs, train_stds, test_accs, test_stds)

    """
    torch.manual_seed(42)
    if not device:
        device = torch.device("cpu" if not torch.cuda.is_available() else "cpu")

    # Extract node features, adjacency matrices, labels and convert to tensors
    x, a, y, num_labels = load_data(path, dataset_name)

    # Create dataset and loader for mini batches
    dataset = TensorDataset(x, a, y)

    # Setup 10-fold cross validation
    kf = KFold(n_splits=fold_count, random_state=42, shuffle=True)

    results = train_accs, train_stds, test_accs, test_stds = ([], [], [], [])
    for fold, (train_indices, test_indices) in enumerate(kf.split(dataset, y)):
        if not QUIET:
            print(f"Running fold {fold + 1}/{fold_count}")

        # Extract training and test data for each fold
        train_data_sub = SubsetRandomSampler(train_indices)
        test_data_sub = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_data_sub,
                                  collate_fn=lambda v: tuple(v_.to(device) for v_ in default_collate(v)))
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_data_sub,
                                 collate_fn=lambda v: tuple(v_.to(device) for v_ in default_collate(v)))

        # Construct neural network and move it to device
        # Input dimension: length of node vectors
        model = GraphLevelGCN(input_dim=len(x[0][0]), output_dim=num_labels, hidden_dim=64)
        model.train()
        model.to(device)

        # Construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.functional.cross_entropy

        # Training loop
        train_acc, train_std = 0, 0
        for epoch in tqdm(range(epochs), disable=QUIET):
            train_acc, train_std = train_loop(train_loader, model, loss_fn, opt)  # Consider only accuracy of last epoch

        train_accs.append(train_acc)
        train_stds.append(train_std)

        test_acc, test_std = test(test_loader, model, loss_fn)
        test_accs.append(test_acc)
        test_stds.append(test_std)

    return results


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Trains the model for one epoch, i.e. on all batches

    Parameters
    ----------
    dataloader The dataloader to use. Needs to provide x: node embeddings, a: normalized adjacency matrices, y: labels
    model The model to train.
    loss_fn The loss function to use. E.g., torch.nn.functional.cross_entropy
    optimizer The optimizer to use. E.g., torch.optim.Adam(model.parameters(), lr=learning_rate)

    Returns
    -------
    The list of accuracy values for each batch.

    """
    accuracies = np.zeros(len(dataloader))
    for batch, (x_train, a_train, y_train) in enumerate(dataloader):
        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass and loss
        y_pred = model(x_train, a_train)
        loss = loss_fn(y_pred, y_train)

        # Backward pass and sgd step
        loss.backward()
        optimizer.step()

        # Record accuracy
        accuracies[batch] = get_accuracy(y_pred, y_train)

    return accuracies.mean(), accuracies.std()  # Means-of-Means can be neglected as data will always be same size


def test(dataloader, model, loss_fn):
    accuracies = np.zeros(len(dataloader))
    with torch.no_grad():
        for batch, (x_test, a_test, y_test) in enumerate(dataloader):
            y_pred = model(x_test, a_test)
            accuracies[batch] = get_accuracy(y_pred, y_test)

    return accuracies.mean(), accuracies.std()  # Means-of-Means can be neglected as data will always be same size


def get_accuracy(pred: Tensor, truth: Tensor) -> float:
    correct = (pred.argmax(1) == truth).type(torch.float).sum().item()
    return correct / len(truth)


if __name__ == '__main__':
    params = {
        "epochs": 10, "batch_size": 32, "learning_rate": 1e-3, "fold_count": 10
    }
    train_accs, train_stds, test_accs, test_stds = run_graph_classification("../Task1/datasets/NCI1/data.pkl", "NCI",
                                                                            device="cuda", **params)
    print(f"Train Accuracy: {np.mean(train_accs)*100:0.2f}%(±{np.mean(train_stds)*100:0.2f})")
    print(f"Test Accuracy: {np.mean(test_accs)*100:0.2f}%(±{np.mean(test_stds)*100:0.2f})")
