import torch
from torch import Tensor

from GCNLayer import GCNLayer


class GMLP(torch.nn.Module):
    """
    A multilevel perceptron working with graph layers.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        super(GMLP, self).__init__()
        self.num_layers = num_layers

        # Add input and output submodules as attributes
        self.input_layer = GCNLayer(input_dim, hidden_dim)
        self.output_layer = GCNLayer(hidden_dim, output_dim)

        # Add hidden layers to module list
        self.hidden_layers = torch.nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        )

    def forward(self, x: Tensor, adj_matrices: Tensor):
        """

        Parameters
        ----------
        x The node feature vectors of each graph.
        adj_matrices Corresponding normalized adjacency matrices of each graph.

        Returns
        -------

        """
        y = self.input_layer(x, adj_matrices)
        for hidden_layer in self.hidden_layers:
            y = hidden_layer(y, adj_matrices)
        y = self.output_layer(y, adj_matrices)
        return y
