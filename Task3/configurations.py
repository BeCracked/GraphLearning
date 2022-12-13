"""
Contains configuration parameter dictionaries that can be used as is or as a basis for new configurations.
"""

zinc_base_params = {
    "node_feature_key": "node_label",
    "edge_feature_key": "edge_label",
    "graph_feature_key": "label",

    "hidden_dim": 45,
    "aggregation": "SUM",
    "drop_prob": 0.001,
    "virtual_node": False,
    "layer_count": 7,

    "node_feature_dimension": 21,
    "edge_feature_dimension": 3,

    "epoch_count": 400,
    "learning_rate": 1e-3,
    "batch_size": 128
}
