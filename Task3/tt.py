params = {
    "a": 5,
    "b": 42,
    "layer_config": {
        "a": 2,
        "b": 43
    }
}


def model(pos, *, a, **config):
    # construct layers
    layers(pos, **config["layer_config"])


def layers(pos, *, a, b, **config):
    print(b)
    pass


if __name__ == '__main__':
    model("pos", **params)




9: best score:  tensor(1.1448, grad_fn=<AbsBackward0>)
with params {'hidden_dim': 10, 'aggregation': 'SUM', 'drop_prob': 0.005, 'virtual_node': False, 'layer_count': 7,
'epoch_count': 20, 'learning_rate': 1e-05, 'batch_size': 128}


9: best score:  tensor(1.0886, grad_fn=<AbsBackward0>)
with params {, 'hidden_dim': 20, 'aggregation': 'SUM', 'drop_prob': 0, 'virtual_node': False, 'layer_count': 7,
 'epoch_count': 20, 'learning_rate': 1e-05, 'batch_size': 128}


jessica: hidden: 50 drop: 0, layers: 8, learn rate -3, batch size:128

9: best score:  tensor(0.2990, grad_fn=<AbsBackward0>) tensor(0.3375, grad_fn=<AbsBackward0>)
tensor(0.3375, grad_fn=<AbsBackward0>) tensor(0.3325, grad_fn=<AbsBackward0>)
with params { 'hidden_dim': 20, 'aggregation': 'SUM', 'drop_prob': 0.005, 'virtual_node': False, 'layer_count': 7,
'epoch_count': 60, 'learning_rate': 0.001, 'batch_size': 128} and 59 epochs

9: best score:  tensor(0.2398, grad_fn=<AbsBackward0>) tensor(0.2914, grad_fn=<AbsBackward0>)
tensor(0.2914, grad_fn=<AbsBackward0>) tensor(0.2725, grad_fn=<AbsBackward0>)
with params { 'hidden_dim': 40, 'aggregation': 'SUM', 'drop_prob': 0.005, 'virtual_node': False, 'layer_count': 7,
 'epoch_count': 60, 'learning_rate': 0.001, 'batch_size': 128} and 57 epochs


