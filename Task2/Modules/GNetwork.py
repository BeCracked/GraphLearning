import torch

from Task2.Modules.GCNLayer import GCNLayer


class GNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        """
        Creates a network of GCN layers.

        Parameters
        ----------
        input_dim Dimension of input layer
        output_dim Dimension of output layer
        hidden_dim Dimension of hidden layers
        num_layers Total number of layers

        Returns
        -------

        """
        super(GNetwork, self).__init__()
        self.num_layers = num_layers

        # Setup input layer
        self.input_layer = GCNLayer(input_dim, hidden_dim)

        # Setup hidden layers in ModuleList
        self.hidden_layers = torch.nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        )

        # Setup output layer
        self.output_layer = GCNLayer(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, adj_matrices: torch.Tensor):
        """
        Forward computation of the GCN Network.

        Parameters
        ----------
        x Input vector of GCN Network
        adj_matrices Adjacency matrices of each graph

        Returns
        -------
        Full GCN Network.
        """

        # apply layers
        y = self.input_layer(x, adj_matrices)
        for i in range(self.num_layers - 2):
            y = self.hidden_layers[i](y, adj_matrices)
        y = self.output_layer(y, adj_matrices)
        return y
