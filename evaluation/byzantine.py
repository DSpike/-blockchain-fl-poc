"""Byzantine attack injectors for robustness evaluation."""

import torch
import copy
import random


def inject_byzantine(client_models, byzantine_fraction, attack_type="random",
                     seed=42, global_model=None):
    """
    Mark a fraction of clients as Byzantine and corrupt their models.

    attack_type:
        "random"          : replace weights with random noise
        "sign_flip"       : flip sign of all model weights (label-flip)
        "scaling"         : scale weights by large factor (boosting)
        "gradient_poison" : preserve local accuracy but randomise gradient
                            direction — fools accuracy-only filters (PoCL)
                            but caught by gradient-divergence filter (Ours)

    "gradient_poison" requires global_model to be passed.
    Returns (list of models, list of byzantine indices).
    """
    rng = random.Random(seed)
    n_byzantine = int(len(client_models) * byzantine_fraction)
    byzantine_ids = rng.sample(range(len(client_models)), n_byzantine)

    # global state dict needed for gradient_poison
    global_sd = global_model.state_dict() if global_model is not None else None

    corrupted = [copy.deepcopy(m) for m in client_models]
    for i in byzantine_ids:
        sd = corrupted[i].state_dict()
        for key in sd:
            if not sd[key].is_floating_point():
                continue
            if attack_type == "random":
                sd[key] = torch.randn_like(sd[key])
            elif attack_type == "sign_flip":
                sd[key] = -sd[key]
            elif attack_type == "scaling":
                sd[key] = sd[key] * 100.0
            elif attack_type == "gradient_poison":
                # Keep model close to global (preserves validation accuracy)
                # but replace the update direction with random noise.
                # Result: accuracy-only filter passes it; gradient-divergence catches it.
                assert global_sd is not None, "gradient_poison requires global_model"
                honest_delta = sd[key] - global_sd[key]
                delta_norm   = honest_delta.norm() + 1e-8
                rand_dir     = torch.randn_like(sd[key])
                rand_dir     = rand_dir / (rand_dir.norm() + 1e-8)
                # Byzantine update = global + random direction with same magnitude
                sd[key] = global_sd[key] + rand_dir * delta_norm
        corrupted[i].load_state_dict(sd)

    return corrupted, byzantine_ids
