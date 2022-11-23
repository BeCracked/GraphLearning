import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from GMLP import GMLP
import preprocessing


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and cast to tensor
    graphs = np.load("../Task1/datasets/ENZYMES/data.pkl", allow_pickle=True)
    x: torch.Tensor = preprocessing.get_node_feature_embeddings(graphs, "node_attributes")
    y = tensor(preprocessing.extract_labels_from_dataset(graphs))
    norm_adj_m = preprocessing.norm_adj_matrices(graphs)

    # Create dataset and loader with batches
    batch_size = 64
    training_dataset = TensorDataset(x, norm_adj_m, y)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    # Construct NN and move it to device
    max_node_count = len(x[0])
    model = GMLP(max_node_count, max_node_count, max_node_count, 3)
    model.train()
    model.to(device)

    # Construct optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        for x, a, y_true in train_loader:
            # Set gradients to zero
            opt.zero_grad()

            # Move data to device
            x = x.to(device)
            a = a.to(device)
            y_true = y_true.to(device)

            # Forward pass and loss
            y_pred = model(x, a)
            loss = torch.nn.functional.cross_entropy(y_pred, y_true)

            # Backward pass and sgd step
            loss.backward()
            opt.step()


if __name__ == "__main__":
    main()
