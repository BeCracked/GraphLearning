import torch

from Modules.sparse_sum_pooling import SparseSumPooling


class VirtualNode(torch.nn.Module):
    def __init__(self, V_dim_in: int, V_dim_out: int):
        """
        Creates a virtual node.

        Parameters
        ----------
        V_dim_in Input dimension of the trainable MLP V.
        V_dim_out Output dimension of the trainable MLP V.

        Returns
        -------

        """
        super().__init__()

        # Setup V as trainable MLP
        self.V = torch.nn.Sequential(
            torch.nn.Linear(V_dim_in, V_dim_out),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(V_dim_out)
        )

        self.sum_pooling = SparseSumPooling()

    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor):
        """
        Forward computation of the virtual node.

        Parameters
        ----------
        h Node feature matrix.
        batch_idx Index of the original graph for each vertex in the merged graph.

        Returns
        -------
        Full virtual node.
        """
        # Apply scatter sum (essentially the sum pooling module)
        sum_features = self.sum_pooling(h, batch_idx)
        # Apply MLP
        h_G = self.V(sum_features)
        # Distribute information
        h_v = torch.index_select(h_G, dim=0, index=batch_idx) + h

        return h_v
