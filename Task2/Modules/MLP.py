import torch

from .NormalLayer import NormalLayer


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        """
        A multilayer perceptron for classification.

        Parameters
        ----------
        input_dim Dimension of input layer
        output_dim Dimension of output layer
        hidden_dim Dimension of hidden layers
        num_layers Total number of layers

        Returns
        -------
        Full MLP for classification.
        """
        super(MLP, self).__init__()
        self.num_layers = num_layers

        # Setup input layer and (linear) output layer
        if self.num_layers > 1:
            self.input_layer = NormalLayer(input_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

        # Setup hidden layers in ModuleList
        if self.num_layers > 1:
            self.hidden_layers = torch.nn.ModuleList(
                [NormalLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
            )

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x Input for classification

        Returns
        -------
        The result of the output layer of this module.

        """
        if self.num_layers > 1:
            y = self.input_layer(x)
            for i in range(self.num_layers - 2):
                y = self.hidden_layers[i](y)
            y = self.output_layer(y)
        else:
            y = self.output_layer(x)
        return y
