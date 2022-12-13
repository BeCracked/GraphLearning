from typing import Literal

import torch
import torch_scatter


# noinspection PyPep8Naming
class SparseGNNLayer(torch.nn.Module):
    def __init__(self, M_dim_in: int, M_dim_out: int, U_dim_in: int, U_dim_out: int, *,
                 aggregation: Literal["SUM", "MEAN", "MAX"], drop_prob: float, **config):
        """
        Creates a GNN layer.

        Parameters
        ----------
        M_dim_in Input dimension of message function M.
        M_dim_out Output dimension of message function M.
        U_dim_in Input dimension of update function U.
        U_dim_out Output dimension of update function U.
        aggregation Type of aggregation ("SUM", "MEAN" or "MAX").
        drob_prob Probability for dropout layer.

        Returns
        -------

        """
        super().__init__()

        self.aggregation = aggregation

        # Setup M and U as trainable MLPs

        self.M = torch.nn.Sequential(
            torch.nn.Linear(M_dim_in, M_dim_out),
            torch.nn.ReLU())

        self.U = torch.nn.Sequential(
            torch.nn.Linear(U_dim_in, U_dim_out),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(U_dim_out)
        )

        # Dropout probability
        self.drop_prob = drop_prob

    def forward(self, H: torch.Tensor, Xe: torch.Tensor, id_Xe: torch.Tensor):
        """
        Forward computation of the GNN layer.

        Parameters
        ----------
        H Node feature matrix.
        Xe Edge feature matrix.
        id_Xe Directed edge list.

        Returns
        -------
        Full GNN layer.
        """
        # Extract features according to indices
        features = torch.index_select(H, dim=0, index=id_Xe[0])
        # Concatenate along feature dimension
        M_con_cat = torch.cat([features, Xe], dim=1)
        # Apply MLP
        Y = self.M(M_con_cat)

        # Apply aggregation
        match self.aggregation:
            case "MEAN":
                Z = torch_scatter.scatter_mean(Y, index=id_Xe[1], dim=0)
            case "MAX":
                Z = torch_scatter.scatter_max(Y, index=id_Xe[1], dim=0)[0]
            case "SUM":
                Z = torch_scatter.scatter_add(Y, index=id_Xe[1], dim=0)

        # Concatenate along feature dimension
        U_con_cat = torch.cat([H, Z], dim=1)
        # Apply MLP
        H_next = self.U(U_con_cat)
        # Apply dropout
        H_drop = torch.nn.functional.dropout(H_next, p=self.drop_prob, training=self.training)
        
        return H_drop
