import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold

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
    node_features = torch.zeros(1)
    if dataset == "NCI":
        node_features = preprocessing.get_node_feature_embeddings(data, False)
    if dataset == "ENZYMES":
        node_features = preprocessing.get_node_feature_embeddings(data, True)

    # Extract labels and cast to tensor
    labels = preprocessing.extract_labels_from_dataset(data)
    possible_labels = list(set(labels))
    num_labels = len(possible_labels)
    # cross entropy takes indices of labels, not labels itself
    labels = [possible_labels.index(label) for label in labels]
    labels = torch.tensor(labels)

    return node_features, norm_matrices, labels, num_labels


def run_graph_classification(path: str, dataset_name: str, device: str = "cpu"):
    # Extract node features, adjacency matrices, labels and convert to tensors
    x, a, y, num_labels = load_data(path, dataset_name)

    # Create dataset and loader for mini batches
    dataset = TensorDataset(x, a, y)

    # TODO: determine epochs and learning and (batch size) rate empirically

    # Setup 10-fold cross validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    train_accuracy = []

    for train_indices, test_indices in kf.split(dataset, y):
        # Extract training and test data for each fold
        train_data_sub = SubsetRandomSampler(train_indices)
        test_data_sub = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_data_sub)
        test_loader = DataLoader(dataset, batch_size=32, sampler=test_data_sub)

        # Construct neural network and move it to device
        # Input dimension: length of node vectors, Output dimension: number of labels
        model = GraphLevelGCN(input_dim=len(x[0][0]), output_dim=num_labels, hidden_dim=64)
        model.train()
        model.to(device)

        # Construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(10):
            for x_train, a_train, y_train in train_loader:
                # Set gradients to zero
                opt.zero_grad()

                # Move data to device
                x_train = x_train.to(device)
                a_train = a_train.to(device)
                y_train = y_train.to(device)
                # Forward pass and loss
                y_pred = model(x_train, a_train)
                loss = torch.nn.functional.cross_entropy(y_pred, y_train)

                # Backward pass and sgd step
                loss.backward()
                opt.step()

        # Evaluate fold on test loader
        num_pred_correct = 0
        num_pred_total = 0
        for x_test, a_test, y_test in test_loader:
            y_pred = model(x_test, a_test)
            num_pred_total += y_test.size(0)
            # we need the predicted class which are the indices where we have entry 1
            _, pred_class = torch.max(y_pred, 1)
            num_pred_correct += torch.sum(pred_class == y_test).item()
        train_accuracy.append(num_pred_correct / num_pred_total)
        print("one fold done")

    test_accuracy = sum(train_accuracy) / len(train_accuracy)

    return train_accuracy, test_accuracy


if __name__ == '__main__':
    train_ac, test_ac = run_graph_classification("../Task1/datasets/NCI1/data.pkl", "NCI", "cpu")
    print(train_ac)
    print(test_ac)
