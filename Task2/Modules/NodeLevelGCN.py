import torch


from .GNetwork import GNetwork
from .MLP import MLP


class NodeLevelGCN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        """
        Construct the node-level GCN according to Exercise 4.

        Parameters
        ----------
        input_dim Dimension of input layer for GCN Network.
        hidden_dim Dimension of hidden layers for GCN Network and MLP Classification.
        output_dim Number of classes for MLP Classification.

        Returns
        -------
        
        """
        super(NodeLevelGCN, self).__init__()

        # Setup Network of 5 GCN layers and hidden dimension of 64
        self.GCNNetwork = GNetwork(input_dim, hidden_dim, hidden_dim, 3)

        # Setup MLP classification (one linear output layer)
        self.MLPClassification = MLP(hidden_dim, output_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj_matrices: torch.Tensor):
        """
        Forward computation of the node-level GCN.

        Parameters
        ----------
        x The node feature vectors of each graph.
        adj_matrices Corresponding normalized adjacency matrices of each graph.

        Returns
        -------
        Full node-level GCN.
        """
        # Apply GCN network
        y = self.GCNNetwork(x, adj_matrices)
        # Add dropout layer to avoid overfitting
        y = torch.nn.functional.dropout(y, p=0.0, training=self.training)
        # Apply MLP classification
        y = self.MLPClassification(y)
        return y
