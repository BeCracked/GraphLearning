import torch
from torch import bmm, Tensor


class GCNLayer(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        """

        Parameters
        ----------
        dim_in The input dimension of this layer.
        dim_out The output dimension of this layer.
        adj_matrices The 3D matrix containing the normalized adjacency matrices.
                     Dimension of b×n×n with n = max(|V|_i) for all G_i.
        """
        super(GCNLayer, self).__init__()

        # Use Kaiming Init when using ReLU # TODO
        self.W = torch.nn.Parameter(torch.empty(dim_in, dim_out))
        torch.nn.init.kaiming_normal_(self.W)

        self.b = torch.nn.Parameter(torch.normal(0.0, 1.0, size=(dim_out,)))

    def forward(self, x: Tensor, adj_matrices: Tensor):
        """
        Forward computation of the layer.

        Parameters
        ----------
        x The node feature vectors of each graph.
        adj_matrices Corresponding normalized adjacency matrices of each graph.

        Returns
        -------

        """
        # Mult adj matrices with input vertex data embedding
        o = bmm(adj_matrices, x)

        # Expand weights vector for batch matrix multiplication
        expand_shape = (o.size()[0], self.W.size()[0], self.W.size()[1])
        w_b = self.W.expand(expand_shape)

        y = bmm(o, w_b)

        # Apply activation function
        y = torch.relu(y)
        return y
