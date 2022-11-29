import torch


from .GNetwork import GNetwork
from .MLP import MLP


class GraphLevelGCN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        """
        Construct the graph-level GCN for according to Exercise 3.

        Parameters
        ----------
        input_dim Dimension of input layer for GCN Network
        hidden_dim Dimension of hidden layers for GCN Network and MLP Classification
        output_dim Number of classes for MLP classification

        Returns
        -------
        Full graph-level GCN.
        """
        super(GraphLevelGCN, self).__init__()

        # Setup Network of 5 GCN layers and hidden dimension of 64
        self.GCNNetwork = GNetwork(input_dim, hidden_dim, hidden_dim, 5)

        # Setup MLP classification (one hidden layer of dimension 64, one linear output layer)
        self.MLPClassification = MLP(hidden_dim, output_dim, hidden_dim, 2)

    def forward(self, x: torch.Tensor, adj_matrices: torch.Tensor):
        # Apply GCN network
        y = self.GCNNetwork(x, adj_matrices)
        # Apply sum pooling
        y = torch.sum(y, 1, keepdim=True)
        # Remove dimensions of 1 (otherwise shape conflict)
        y = torch.squeeze(y)
        # Add dropout layer to avoid overfitting
        y = torch.nn.functional.dropout(y, p=0.0, training=self.training)
        # Apply MLP classification
        y = self.MLPClassification(y)
        return y
