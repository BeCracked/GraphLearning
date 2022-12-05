import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx


class SparseGraphDataset(Dataset):
    def __init__(self, graphs: list[nx.Graph], *,
                 device: str | None = None,
                 node_feature_key: str | None = "node_label", edge_feature_key: str | None = None,
                 graph_feature_key: str | None = "label",
                 **kwargs):
        super(SparseGraphDataset, self).__init__()
        self.graph_count = len(graphs)

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.edge_lists: list = [None] * self.graph_count
        self.node_features: list = [None] * self.graph_count
        self.edge_features: list = [None] * self.graph_count
        self.graph_labels: list = [None] * self.graph_count

        for i, g in enumerate(graphs):
            i: int
            # Node features
            if node_feature_key:
                self.node_features[i] = torch.tensor([val for node, val in g.nodes(data=node_feature_key)])

            # Directed Edge List
            s, t, v = map(list, zip(*g.to_directed().edges(data=edge_feature_key)))  # source, target, value
            idx = torch.tensor([s, t])
            shape = (2, len(v))
            self.edge_lists[i] = idx

            if edge_feature_key:
                self.edge_features[i] = torch.tensor(v).to_sparse_coo()

            # Graph Labels
            if graph_feature_key:
                self.graph_labels[i] = g.graph[graph_feature_key]

    def __getitem__(self, item):
        return self.edge_lists[item], self.node_features[item], self.edge_features[item], self.graph_labels[item]

    def __len__(self):
        return self.graph_count


def sparse_graph_collation(datasets: list[SparseGraphDataset]):
    raise NotImplementedError


if __name__ == '__main__':
    path = "../datasets/ZINC_Train/data.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f)

    dataset = SparseGraphDataset(data, edge_feature_key="edge_label")
    dl = DataLoader(dataset, collate_fn=sparse_graph_collation)
