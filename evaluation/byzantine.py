"""Byzantine attack injectors for robustness evaluation."""

import torch
import copy
import random


def inject_byzantine(client_models, byzantine_fraction, attack_type="random", seed=42):
    """
    Mark a fraction of clients as Byzantine and corrupt their models.

    attack_type:
        "random"     : replace weights with random noise
        "sign_flip"  : flip sign of all gradients (label-flip attack)
        "scaling"    : scale updates by large factor (boosting attack)

    Returns list of (possibly corrupted) models.
    """
    rng = random.Random(seed)
    n_byzantine = int(len(client_models) * byzantine_fraction)
    byzantine_ids = rng.sample(range(len(client_models)), n_byzantine)

    corrupted = [copy.deepcopy(m) for m in client_models]
    for i in byzantine_ids:
        sd = corrupted[i].state_dict()
        for key in sd:
            if attack_type == "random":
                sd[key] = torch.randn_like(sd[key])
            elif attack_type == "sign_flip":
                sd[key] = -sd[key]
            elif attack_type == "scaling":
                sd[key] = sd[key] * 100.0
        corrupted[i].load_state_dict(sd)

    return corrupted, byzantine_ids
