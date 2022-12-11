import torch
from Task3.Modules.RNetwork import RNetwork


class GraphRegressionGCN(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, agg_type: str, v_nodes: bool,
                 drop_prob: int, node_feature_dimension: int, edge_feature_dimension: int):
        """
        Construct the graph-regression GCN according to Exercise 3.6.

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
        v_nodes Should virtual nodes be used (yes or no).


        Returns
        -------

        """
        super(GraphRegressionGCN, self).__init__()

        self.GNN = RNetwork(hidden_dim=hidden_dim, agg_type=agg_type, virtual_node=v_nodes, drop_prob=drop_prob,
                            node_feature_dimension=node_feature_dimension, edge_feature_dimension=edge_feature_dimension,
                            layer_count=num_layers)

    def forward(self, x: torch.Tensor, adj_matrices: torch.Tensor):
        """
        Forward computation of the graph-regression GCN.

        Parameters
        ----------
        x The node feature vectors of each graph.
        adj_matrices Corresponding normalized adjacency matrices of each graph.

        Returns
        -------
        Full graph-level GCN.
        """
        # Apply GCN network
        y = self.GCNNetwork(x, adj_matrices)
        # Apply sum pooling
        y = torch.sum(y, 1, keepdim=True)
        # Remove dimensions of 1 (otherwise shape conflict)
        y = torch.squeeze(y)
        # Add dropout layer to avoid overfitting
        y = torch.nn.functional.dropout(y, p=0.0, training=self.training)
        # Apply MLP classification
        y = self.MLPClassification(y)
        return y
