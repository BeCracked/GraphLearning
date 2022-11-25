import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset

import preprocessing
from GraphLevelGCN import GraphLevelGCN


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
    node_features = preprocessing.get_node_feature_embeddings(data)

    # Extract labels, apply one and cast to tensor
    labels = preprocessing.extract_labels_from_dataset(data)
    num_labels = len(set(labels))
    labels = torch.tensor(labels)
    if dataset == 'NCI':
        labels = torch.nn.functional.one_hot(labels, num_classes=num_labels)

    return node_features, norm_matrices, labels, num_labels


def run_graph_classification(device='cpu'):
    # Extract node features, adjacency matrices, labels and convert to tensors
    x, a, y, num_labels = load_data("../Task1/datasets/ENZYMES/data.pkl")

    # Create dataset and loader for mini batches
    train_dataset = TensorDataset(x, a, y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Construct neural network and move it to device
    # Input dimension: length of node vectors, Output dimension: number of labels
    model = GraphLevelGCN(input_dim=len(x[0][0]), output_dim=num_labels, hidden_dim=64)
    model.train()
    model.to(device)

    # TODO: determine epochs and learning rate empirically
    # construct optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
        for x_train, a_train, y_true in train_loader:
            # Set gradients to zero
            opt.zero_grad()

            # Move data to device
            x_train = x_train.to(device)
            a_train = a_train.to(device)
            y_true = y_true.to(device)

            # Forward pass and loss
            y_pred = model(x_train, a_train)
            loss = torch.nn.functional.cross_entropy(y_pred, y_true)

            # Backward pass and sgd step
            loss.backward()
            opt.step()


if __name__ == '__main__':
    run_graph_classification()

