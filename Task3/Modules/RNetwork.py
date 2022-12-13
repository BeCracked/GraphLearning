import torch
import torch_scatter


from Modules.sparse_gnn_layer import SparseGNNLayer
from Modules.virtual_node import VirtualNode
from Modules.sparse_sum_pooling import SparseSumPooling


class RNetwork(torch.nn.Module):
    def __init__(self, node_feature_dimension: int, edge_feature_dimension: int, *,
                 hidden_dim: int, virtual_node: bool, layer_count: int, drop_prob: float = 0, **config):
        """
        Creates a network of GNN layers with optional virtual node, global pooling and final MLP.

        Parameters
        ----------
        node_feature_dimension Length of node features (21 in our case).
        edge_feature_dimension Length of edge features (3 in our case).
        hidden_dim Hidden dimension that can be optimized.
        virtual_node Flag whether to use virtual nodes or not.
        layer_count Number of GNN layers.
        drop_prob Probability for dropout layers.
        config Configuration dictionary that may be used in other modules or functions.

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
        Forward computation of the network.

        Parameters
        ----------
        H Node feature matrix.
        Xe Edge feature matrix.
        id_Xe Directed edge list.
        batch_idx Index of the original graph for each vertex in the merged graph.

        Returns
        -------
        Full regression network.
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

        # apply MLP
        y = self.MLP(y)

        return y
