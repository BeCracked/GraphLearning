from typing import Literal

import torch
import torch_scatter


# noinspection PyPep8Naming
class SparseGNNLayer(torch.nn.Module):
    def __init__(self, M_dim_in: int, M_dim_out: int, U_dim_in: int, U_dim_out: int, *,
                 aggregation: Literal["SUM", "MEAN", "MAX"], drop_prob: float, **config):
        super().__init__()

        self.aggregation = aggregation

        # Setup M and U as trainable MLPs TODO: Apply MLP? Do we need hidden layers?

        self.M = torch.nn.Sequential(
            torch.nn.Linear(M_dim_in, M_dim_out),
            torch.nn.ReLU())

        self.U = torch.nn.Sequential(
            torch.nn.Linear(U_dim_in, U_dim_out),
            torch.nn.ReLU())

        # Dropout probability
        self.drop_prob = drop_prob

    def forward(self, H: torch.Tensor, Xe: torch.Tensor, id_Xe: torch.Tensor):
        # Extract features according to indices
        features = torch.index_select(H, dim=0, index=id_Xe[0])
        # Concatenate along feature dimension
        M_con_cat = torch.cat([features, Xe], dim=1)
        # Apply MLP
        Y = self.M(M_con_cat)

        # apply aggregation
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
        # TODO: apply global pooling by using virtual nodes if the parameters say so
        return H_drop
