import torch
import GCNLayer


class GCNGraph(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=5):
        """
        GCN graph setup: creates input, output and hidden layers according to dimensionality.

        Parameters
        ----------
        input_dim Dimension of input layer
        output_dim Dimension of output layer
        hidden_dim Dimension of hidden layers (we use 64)
        num_layers Total number of layer in GCN graph (we use 5)

        Returns
        -------

        """
        super(GCNGraph, self).__init__()
        self.num_layers = num_layers

        # setup input layer and output layer
        self.input_layer = GCNLayer.GCNLayer(input_dim, hidden_dim)
        self.output_layer = GCNLayer.GCNLayer(hidden_dim, output_dim)

        # setup hidden layers in ModuleList
        self.hidden_layers = torch.nn.ModuleList(
            [GCNLayer.GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        )

    def forward(self, x):
        """
        Forward computation of the GCN graph.

        Parameters
        ----------
        x Input vector of GCN graph

        Returns
        -------
        Output vector of GCN graph
        """

        y = self.input_layer(x)
        for i in range(self.num_layers - 2):
            y = self.hidden_layers[i](y)
        y = self.output_layer(y)
        return y
