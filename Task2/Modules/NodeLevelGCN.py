import torch


from Task2.Modules.GNetwork import GNetwork
from Task2.Modules.MLP import MLP


class NodeLevelGCN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        """
        Construct the node-level GCN for according to Exercise 4.

        Parameters
        ----------
        input_dim Dimension of input layer for GCN Network
        hidden_dim Dimension of hidden layers for GCN Network and MLP Classification
        output_dim Number of classes for MLP classification

        Returns
        -------
        Full node-level GCN.
        """
        super(NodeLevelGCN, self).__init__()

        # Setup Network of 5 GCN layers and hidden dimension of 64
        self.GCNNetwork = GNetwork(input_dim, hidden_dim, hidden_dim, 3)

        # Setup MLP classification (one linear output layer)
        self.MLPClassification = MLP(hidden_dim, output_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj_matrices: torch.Tensor):
        # Apply GCN network
        y = self.GCNNetwork(x, adj_matrices)
        # Apply MLP classification
        y = self.MLPClassification(y)
        return y
