import torch
import torch_scatter


from Modules.sparse_gnn_layer import SparseGNNLayer
from Modules.virtual_node import VirtualNode
from Modules.sparse_sum_pooling import SparseSumPooling


# noinspection PyPep8Naming
class RNetwork(torch.nn.Module):
    def __init__(self, node_feature_dimension: int, edge_feature_dimension: int, *,
                 hidden_dim: int, virtual_node: bool, layer_count: int, drop_prob: float = 0, **config):
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
        virtual_nodes Should virtual nodes be used (yes or no).

        Returns
        -------

        """

        super(RNetwork, self).__init__()
        self.num_layers = layer_count
        self.virtual_node = virtual_node
        self.drop_prob = drop_prob

        # Setup input layer
        self.input_layer = SparseGNNLayer(node_feature_dimension + edge_feature_dimension, hidden_dim,
                                          hidden_dim+node_feature_dimension, hidden_dim,
                                          drop_prob=drop_prob, **config)

        # Setup hidden layers in ModuleList
        self.hidden_layers = torch.nn.ModuleList(
            [SparseGNNLayer(hidden_dim+edge_feature_dimension, hidden_dim,
                            2*hidden_dim, hidden_dim,
                            drop_prob=drop_prob, **config)
             for _ in range(layer_count - 1)]
        )

        if self.virtual_node:
            self.virtual_nodes = torch.nn.ModuleList(
                [VirtualNode(hidden_dim, hidden_dim) for _ in range(layer_count - 1)]
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
        if self.virtual_node:
            y = self.virtual_nodes[0](y, batch_idx)

        for i in range(self.num_layers - 1):
            y = self.hidden_layers[i](y, Xe, id_Xe)
            # do not add virtual node after last layer
            if self.virtual_node and i < self.num_layers - 2:
                y = self.virtual_nodes[i + 1](y, batch_idx)
        y = self.global_pool(y, batch_idx)

        # Apply MLP Classification
        y = self.MLP(y)

        return y
