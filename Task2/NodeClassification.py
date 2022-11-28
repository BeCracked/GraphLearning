import numpy as np
import torch
import pickle

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import preprocessing
from NodeLevelGCN import NodeLevelGCN


def load_data(path: str):
    """
    Load graphs of dataset, apply normalization to adjacency matrices and extract labels.
    ----------
    path Path to dataset

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
    node_features = preprocessing.get_node_feature_embeddings(data, True)
    # Extract labels and cast to tensor
    labels = preprocessing.extract_node_labels_from_dataset(data, "node_label")
    possible_labels = list(set(labels))
    num_labels = len(possible_labels)
    # cross entropy takes indices of labels, not labels itself
    labels = [possible_labels.index(label) for label in labels]
    labels = [labels]
    labels = torch.tensor(labels)

    return node_features, norm_matrices, labels, num_labels


def run_node_classification(path_train: str, path_test, *,
                             device: str = "cpu",
                             epochs=10, learning_rate=1e-3) \
        -> tuple[list, list, list, list]:
    """
    Repeats training and testing 10 times with the node classification net on the given training and test dataset.

    Parameters
    ----------
    path_train The path to the training dataset file.
    path_test The path to the test dataset file.
    device The device name to run on.
    epochs Number of epochs to train for.
    learning_rate Learning rate to use by the optimizer.

    Returns
    -------
    The accuracy results for training and test run.
    Tuple of Lists of the form (train_accs, train_stds, test_accs, test_stds)

    """
    # Extract node features, adjacency matrices, labels and convert to tensors
    x_train, a_train, y_train, num_labels_train = load_data(path_train)
    x_test, a_test, y_test, num_labels_test = load_data(path_test)

    # Create dataset and loader for mini batches
    dataset_train = TensorDataset(x_train, a_train, y_train)
    dataset_test = TensorDataset(x_test, a_test, y_test)

    results = train_accs, train_stds, test_accs, test_stds = ([], [], [], [])
    # repeat training and testing 10 times
    for i in range(10):
        print(f"Running fold {i + 1}/{10}")

        train_loader = DataLoader(dataset_train, batch_size=1,
                                  collate_fn=lambda v: tuple(v_.to(device) for v_ in default_collate(v)))
        test_loader = DataLoader(dataset_test, batch_size=1,
                                 collate_fn=lambda v: tuple(v_.to(device) for v_ in default_collate(v)))

        # Construct neural network and move it to device
        # Input dimension: length of node vectors
        model = NodeLevelGCN(input_dim=len(x_train[0][0]), output_dim=num_labels_train, hidden_dim=64)
        model.train()
        model.to(device)

        # Construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.functional.cross_entropy

        # Training loop
        train_acc, train_std = 0, 0
        for epoch in tqdm(range(epochs)):
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

        # To transform it into suitable shape for cross entropy
        y_pred = torch.squeeze(y_pred)
        y_train = torch.squeeze(y_train)

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

            # To transform it into suitable shape for cross entropy
            y_pred = torch.squeeze(y_pred)
            y_test = torch.squeeze(y_test)

            accuracies[batch] = get_accuracy(y_pred, y_test)

    return accuracies.mean(), accuracies.std()  # Means-of-Means can be neglected as data will always be same size


def get_accuracy(pred: Tensor, truth: Tensor) -> float:
    correct = (pred.argmax(1) == truth).type(torch.float).sum().item()
    return correct / len(truth)


if __name__ == '__main__':
    params = {
        "epochs": 10, "learning_rate": 1e-3
    }
    res = run_node_classification("datasets/Citeseer_Train/data.pkl", "datasets/Citeseer_Eval/data.pkl", device="cpu", **params)

    print(res)


# if __name__ == '__main__':
#     with open("datasets/Citeseer_Train/data.pkl", "rb") as f:
#         data = pickle.load(f)
#     node_attributes = data[0].nodes(data=True)[1]
#     print(node_attributes)
