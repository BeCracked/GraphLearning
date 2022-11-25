import torch


from GNetwork import GNetwork
from GMLP import GMLP


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

        # Setup MLP classification (one hidden layer of dimension 64, three layers in total)
        self.MLPClassification = GMLP(hidden_dim, output_dim, hidden_dim, 3)

    def forward(self, x: torch.Tensor, adj_matrices: torch.Tensor):
        # Apply GCN network
        y = self.GCNNetwork(x, adj_matrices)
        # Apply sum pooling # TODO: not sure if pooling is correct
        y = torch.sum(y, 1, keepdim=True)
        # Apply MLP classification
        y = self.MLPClassification(y)
        return y
