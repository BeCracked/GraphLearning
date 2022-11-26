import torch


class NormalLayer(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        """
        Used for the MLP classification as hidden layer.

        Parameters
        ----------
        dim_in The input dimension of this layer.
        dim_out The output dimension of this layer.

        Returns
        -------

        """
        super(NormalLayer, self).__init__()

        # Use Kaiming Init when using ReLU # TODO
        self.W = torch.nn.Parameter(torch.empty(dim_in, dim_out))
        torch.nn.init.kaiming_normal_(self.W)

        self.b = torch.nn.Parameter(torch.normal(0.0, 1.0, size=(dim_out,)))

    def forward(self, x: torch.Tensor):
        """
        Forward computation of the layer.

        Parameters
        ----------
        x The node feature vectors of each graph.

        Returns
        -------

        """
        # Apply layer
        y = torch.matmul(x, self.W) + self.b

        # Apply activation function
        y = torch.relu(y)
        return y
