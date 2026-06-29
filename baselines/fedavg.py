"""Standard FedAVG baseline (uniform weights, no PoC, no blockchain)."""

import torch
import copy


def uniform_fedavg(global_model, client_models):
    n = len(client_models)
    weights = [1.0 / n] * n
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        for model, w in zip(client_models, weights):
            global_state[key] += w * model.state_dict()[key].float()
    global_model.load_state_dict(global_state)
    return global_model
