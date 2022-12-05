import pickle
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx

from collections import namedtuple

SparseGraphRep = namedtuple(
    "SparseGraphRep",
    ["edge_list", "node_features", "edge_features", "graph_label"]
)


class SparseGraphDataset(Dataset):
    def __init__(self, graphs: list[nx.Graph], *,
                 device: str | None = None,
                 node_feature_key: str = "node_label", edge_feature_key: Optional[str] = None,
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
                self.node_features[i] = torch.tensor(
                    [val for node, val in g.nodes(data=node_feature_key)], device=device)

            # Directed Edge List
            s, t, v = map(list, zip(*g.to_directed().edges(data=edge_feature_key)))  # source, target, value
            idx = torch.tensor([s, t], device=device)
            self.edge_lists[i] = idx

            if edge_feature_key:
                self.edge_features[i] = torch.tensor(v, device=device).to_sparse_coo()

            # Graph Labels
            if graph_feature_key:
                self.graph_labels[i] = torch.tensor(g.graph[graph_feature_key])

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return SparseGraphRep(self.edge_lists[item], self.node_features[item],
                              self.edge_features[item], self.graph_labels[item])

    def __len__(self) -> int:
        return self.graph_count


def sparse_graph_collation(sparse_reps: list[SparseGraphRep]):
    edge_features_flag: bool = sparse_reps[0].edge_features is not None and sparse_reps[0].edge_features._nnz() > 0

    edge_list = torch.cat([r.edge_list for r in sparse_reps], dim=1)
    node_features = torch.cat([r.node_features for r in sparse_reps])
    edge_features = None
    if edge_features_flag:
        edge_features = torch.cat([r.edge_features for r in sparse_reps])
    graph_labels = torch.cat([r.graph_label for r in sparse_reps])

    total_node_count = sum((len(r.node_features) for r in sparse_reps))
    batch_idx = torch.zeros(total_node_count, dtype=torch.int)
    b_i = 0  # Last position in the batch_idx vector

    for g_idx, rep in enumerate(sparse_reps):
        start, end = b_i, b_i + rep.node_features.shape[0]
        for i in range(start, end):
            batch_idx[i] = int(g_idx)
        b_i = end

    collated_rep = SparseGraphRep(edge_list, node_features, edge_features, graph_labels)

    return batch_idx, *collated_rep


if __name__ == '__main__':
    import time
    from Task3.configurations import zinc_base_params
    from Task3.DataHandling.preprocessing import edge_labels_to_one_hot
    s = time.perf_counter()
    path = "../datasets/ZINC_Train/data.pkl"
    with open(path, 'rb') as f:
        data = edge_labels_to_one_hot(pickle.load(f))

    dataset = SparseGraphDataset(data, **zinc_base_params)
    reps = []
    for rp in dataset:
        reps.append(rp)
    sparse_graph_collation(reps)
    dl = DataLoader(dataset, collate_fn=sparse_graph_collation)

    e = time.perf_counter()

    print(f"Took {e-s}")
    print(dl)
