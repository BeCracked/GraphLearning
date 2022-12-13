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
        """
        Custom dataset class that constructs a sparse representation of each graph in the dataset.

        Parameters
        ----------
        graphs List of networkx graphs.
        device Which device to use ("cuda" or "cpu").
        node_feature_key Keyword to extract node labels.
        edge_feature_key Keyword to extract edge labels.
        graph_feature_key Keyword to extract graph labels.
        
        Returns
        -------
        
        """
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
                self.edge_features[i] = torch.tensor(v, device=device)

            # Graph Labels
            if graph_feature_key:
                self.graph_labels[i] = torch.tensor(g.graph[graph_feature_key])

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return SparseGraphRep(self.edge_lists[item], self.node_features[item],
                              self.edge_features[item], self.graph_labels[item])

    def __len__(self) -> int:
        return self.graph_count


def sparse_graph_collation(sparse_reps: list[SparseGraphRep]):
    """
    Merges the list of sparse representation graphs into one graph.

    Parameters
    ----------
    sparse_reps List of sparse representation graphs to be merged.
    
    Returns
    -------
    batch_idx Stores the index of the original graph for each vertex in the merged graph.
    collated_rep Contains the directed edge list, node feature matrix, edge feature matrix and graph label.
    """
    edge_features_flag = True

    node_features = torch.cat([r.node_features for r in sparse_reps])
    # Index of node in node_features
    edge_list = sparse_reps[0].edge_list
    offset = len(sparse_reps[0].node_features)
    for rep in sparse_reps[1:]:
        # Get offset for node indices
        partial_edge_list = torch.add(rep.edge_list, offset)
        edge_list = torch.cat([edge_list, partial_edge_list], dim=1)
        offset += len(rep.node_features)

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

    node_features = node_features.float()
    edge_features = edge_features.float()
    batch_idx = batch_idx.long()

    collated_rep = SparseGraphRep(edge_list, node_features, edge_features, graph_labels)

    return batch_idx, *collated_rep
