import torch


from GNetwork import GNetwork
from GMLP import GMLP


class GraphLevelGCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        """
        Construct the graph-level GCN for according to Exercise 3.

        Parameters
        ----------
        input_dim Dimension of input layer for GCN Network
        hidden_dim Dimension of hidden layer for GCN Network
        num_classes Number of classes for MLP classification

        Returns
        -------
        Full graph-level GCN.
        """
        super(GraphLevelGCN, self).__init__()

        # Setup Network of 5 GCN layers and hidden dimension of 64
        self.GCNNetwork = GNetwork(input_dim, hidden_dim, 64, 5)

        # Setup MLP classification (one hidden layer of dimension 64, three layers in total)
        self.MLPClassification = GMLP(hidden_dim, num_classes, 64, 3)

    def forward(self, x: torch.Tensor, adj_matrices: torch.Tensor):
        # Apply GCN network
        y = self.GCNNetwork(x, adj_matrices)
        # Apply sum pooling # TODO: not sure if pooling is correct
        y = torch.sum(y, 1, keepdim=True)
        # Apply MLP classification
        y = self.MLPClassification(y)
        return y
