"""
Contains configuration parameter dictionaries that can be used as is or as a basis for new configurations.
"""

zinc_base_params = {
    "node_feature_key": "node_label",
    "edge_feature_key": "edge_label",
    "graph_feature_key": "label",

    "hidden_dim": 64,
    "aggregation": "SUM",
    "drop_out": 0.05,
    "virtual_node": True,
    "layer_count": 4,

}
