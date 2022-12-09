import torch

from .GCNLayer import GCNLayer
from sparse_gnn_layer import SparseGNNLayer


class RNetwork(torch.nn.Module):
    def __init__(self, int, m_dim_out: int, u_dim_in: int, u_dim_out: int,input_dim: int, output_dim: int,
                 hidden_dim: int, num_layers: int, agg_type: str, v_nodes: bool, drop_prob: int):
        """
        Creates a network of GCN layers.

        Parameters
        ----------
        m_dim_in Input dimension of message function M.
        m_dim_out Output dimension of message function M.
        u_dim_in Input dimension of update function U.
        u_dim_out Output dimension of update function U.
        input_dim Dimension of input layer.
        output_dim Dimension of output layer.
        hidden_dim Dimension of hidden layers.
        depth Total number of layers.
        agg_type Type of aggregation function used ("SUM", "MEAN", or "MAX").
        v_nodes Should virutal nodes be used (yes or no).

        Returns
        -------

        """

        super(RNetwork, self).__init__()
        self.num_layers = num_layers

        # Setup input layer
        self.input_layer = GCNLayer(input_dim, hidden_dim)

        # Setup hidden layers in ModuleList
        self.hidden_layers = torch.nn.ModuleList(
            [SparseGNNLayer(M_dim_in, M_dim_out, U_dim_in, U_dim_out, agg_type, drop_prob) for _ in range(num_layers - 2)]
        )

        # Setup output layer
        self.output_layer = SparseGNNLayer(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, adj_matrices: torch.Tensor):
        """
        Forward computation of the GCN Network.

        Parameters
        ----------
        x The node feature vectors of each graph.
        adj_matrices Corresponding normalized adjacency matrices of each graph.

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
