import torch
from torch.utils.data import Dataset
import networkx as nx


class SparseGraphDataset(Dataset):
    def __init__(self, graphs: list[nx.Graph]):
        super(SparseGraphDataset, self).__init__()
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self, item):
        raise NotImplementedError


def sparse_graph_collation(datasets: list[SparseGraphDataset]):
    raise NotImplementedError
