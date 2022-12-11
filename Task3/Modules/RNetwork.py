import torch


from sparse_gnn_layer import SparseGNNLayer
from virtual_node import VirtualNode
from sparse_sum_pooling import SparseSumPooling


class RNetwork(torch.nn.Module):
    def __init__(self, int, m_dim_out: int, u_dim_in: int, u_dim_out: int,  input_dim: int, output_dim: int,
                 hidden_dim: int, num_layers: int, agg_type: str, v_nodes: bool, drop_prob: int,
                 node_feature_dimension: int, edge_feature_dimension: int, number_of_nodes: int, number_of_edges: int):
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
        self.v_nodes = v_nodes

        # Setup input layer
        # TODO: verify, if input layer is normal gcn layer of sparse gnn layer
        self.input_layer = SparseGNNLayer(node_feature_dimension + edge_feature_dimension, hidden_dim,
                                          hidden_dim+node_feature_dimension, hidden_dim, agg_type, drop_prob)

        # Setup hidden layers in ModuleList
        self.hidden_layers = torch.nn.ModuleList(
            [SparseGNNLayer(hidden_dim+edge_feature_dimension, hidden_dim, 2*hidden_dim, hidden_dim, agg_type,
                            drop_prob) for _ in range(num_layers - 1)]
        )

        if self.v_nodes:
            self.virtual_node = torch.nn.ModuleList(
                [VirtualNode(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
            )

        self.global_pool = SparseSumPooling()

        self.MLP = torch.nn.Linear(hidden_dim, 1)

    def forward(self, H: torch.Tensor, Xe: torch.Tensor, id_Xe: torch.Tensor, batch_idx: torch.Tensor):
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

        y = self.input_layer(H, Xe, id_Xe)
        if self.v_nodes:
            y = self.virtual_node(y, batch_idx)

        for i in range(self.num_layers - 1):
            y = self.hidden_layers[i](y, Xe, id_Xe)
            if self.v_nodes:
                y = self.virtual_node(y, batch_idx)
        y = self.global_pool(y, batch_idx)

        y = self.MLP(y)

        return y
