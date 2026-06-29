"""Weighted FedAVG aggregator using PoC contribution scores."""

import copy
import torch


def weighted_fedavg(global_model, client_models, weights):
    """
    Aggregate client models using contribution-based weights.

    Args:
        global_model  : current global model (for structure reference)
        client_models : list of local models
        weights       : np.array of normalised contribution scores (sum=1)

    Returns updated global model (in-place).
    """
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        for model, w in zip(client_models, weights):
            global_state[key] += w * model.state_dict()[key].float()
    global_model.load_state_dict(global_state)
    return global_model
