import torch
import torch_scatter


class SparseSumPooling(torch.nn.Module):
    def __init__(self):
        """
        Creates the sum pooling layer for the network.

        Parameters
        ----------

        Returns
        -------

        """
        super().__init__()

    def forward(self, H: torch.Tensor, batch_idx: torch.Tensor):
        """
        Forward computation of the sum pooling layer.

        Parameters
        ----------
        H Node feature matrix.
        batch_idx Index of the original graph for each vertex in the merged graph.

        Returns
        -------
        Pooled node feature matrix.
        """
        return torch_scatter.scatter(H, index=batch_idx, dim=0, reduce='sum')
