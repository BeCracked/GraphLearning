import torch
import torch_scatter


class SparseSumPooling(torch.Module):
    def __init__(self):
        super().__init__()

    def forward(self, H: torch.Tensor, batch_idx: torch.Tensor):
        # TODO: is that all?
        return torch_scatter.scatter(H, index=batch_idx, dim=0, reduce='sum')
