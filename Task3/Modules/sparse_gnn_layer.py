from typing import Literal

import torch
import torch_scatter


class SparseGNNLayer(torch.nn.Module):
    def __init__(self, M_dim_in: int, M_dim_out: int, U_dim_in: int, U_dim_out: int, *, aggregation: Literal["SUM", "MEAN", "MAX"] | str):
        super().__init__()

        self.aggregation = aggregation

        # Setup M and U as trainable MLPs TODO: Apply MLP? Do we need hidden layers?
        self.M = torch.nn.Sequential(
            torch.nn.Linear(M_dim_in, M_dim_out),
            torch.nn.ReLU()
        )

        self.U = torch.nn.Sequential(
            torch.nn.Linear(U_dim_in, U_dim_out),
            torch.nn.ReLU()
        )

    def forward(self, H: torch.Tensor, Xe: torch.Tensor, id_Xe: torch.Tensor):
        # Extract features according to indices
        features = torch.index_select(H, dim=0, index=id_Xe[0])
        # Concatenate along feature dimension
        M_con_cat = torch.cat([features, Xe], dim=1)
        # Apply MLP
        Y = self.M(M_con_cat)

        # apply aggregation
        match self.aggregation:
            case "SUM":
                Z = torch_scatter.scatter_add(Y, index=id_Xe[1], dim=0)
            case "MEAN":
                Z = torch_scatter.scatter_mean(Y, index=id_Xe[1], dim=0)
            case "MAX":
                Z = torch_scatter.scatter_max(Y, index=id_Xe[1], dim=0)[0]

        # Concatenate along feature dimension
        U_con_cat = torch.cat([H, Z], dim=1)
        # Apply MLP
        H_next = self.U(U_con_cat)

        return H_next
