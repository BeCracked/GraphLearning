from typing import Literal

import torch


class SparseGNNLayer(torch.Module):
    def __init__(self, dim_in: int, dim_out: int, *, aggregation: Literal["SUM", "MEAN", "MAX"] | str):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return x
