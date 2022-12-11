import torch

from Task3.Modules.sparse_sum_pooling import SparseSumPooling


class VirtualNode(torch.nn.Module):
    def __init__(self, V_dim_in: int, V_dim_out: int):
        super().__init__()

        # Setup V as trainable MLP
        self.V = torch.nn.Sequential(
            torch.nn.Linear(V_dim_in, V_dim_out),
            torch.nn.ReLU()
        )

        self.sum_pooling = SparseSumPooling()

    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor):
        # Apply sum pooling
        sum_features = self.sum_pooling(h, batch_idx)
        # Apply MLP
        h_G = self.V(sum_features)
        # Distribute information
        h_v = torch.index_select(h_G, dim=0, index=batch_idx) + h

        return h_v
