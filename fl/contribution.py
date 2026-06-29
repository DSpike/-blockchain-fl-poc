"""
Non-IID-aware Proof of Contribution score.

contribution_i = alpha * val_acc_i + beta * grad_div_i

  val_acc_i  : accuracy of client i's model on shared validation set
  grad_div_i : 1 - cosine_similarity(grad_i, mean_grad)
               captures how unique client i's update direction is,
               which correlates with rare-class data under non-IID

This is the core novelty of the paper.
"""

import torch
import torch.nn.functional as F
import numpy as np


def flatten_params(model):
    return torch.cat([p.detach().cpu().flatten() for p in model.parameters()])


def gradient_divergence(local_grad: torch.Tensor, mean_grad: torch.Tensor) -> float:
    """
    1 - cosine_similarity(local_grad, mean_grad).
    Range [0, 2]; higher = more unique direction relative to the average.
    Normalised to [0, 1] by dividing by 2.
    """
    cos_sim = F.cosine_similarity(local_grad.unsqueeze(0),
                                  mean_grad.unsqueeze(0)).item()
    return (1.0 - cos_sim) / 2.0


def evaluate_accuracy(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def compute_contribution_scores(client_models, global_model, val_loader,
                                 device, alpha=0.5, beta=0.5):
    """
    Args:
        client_models : list of trained local models (one per client)
        global_model  : current global model (before aggregation)
        val_loader    : shared validation DataLoader
        device        : torch device
        alpha, beta   : weighting coefficients (alpha + beta = 1 recommended)

    Returns:
        scores        : np.array of shape (num_clients,), normalised to sum=1
        raw           : dict with val_acc and grad_div per client (for logging)
    """
    global_vec = flatten_params(global_model)
    grads = [flatten_params(m) - global_vec for m in client_models]
    mean_grad = torch.stack(grads).mean(dim=0)

    val_accs, grad_divs = [], []
    for i, model in enumerate(client_models):
        val_accs.append(evaluate_accuracy(model, val_loader, device))
        grad_divs.append(gradient_divergence(grads[i], mean_grad))

    val_accs  = np.array(val_accs)
    grad_divs = np.array(grad_divs)

    raw_scores = alpha * val_accs + beta * grad_divs
    # normalise so weights sum to 1 for weighted FedAVG
    normalised = raw_scores / (raw_scores.sum() + 1e-9)

    return normalised, {"val_acc": val_accs, "grad_div": grad_divs, "raw": raw_scores}


def filter_byzantine(scores, raw, threshold_std=2.0):
    """
    Exclude clients whose raw contribution score is > threshold_std below mean.
    Returns boolean mask (True = keep).
    """
    raw_scores = raw["raw"]
    cutoff = raw_scores.mean() - threshold_std * raw_scores.std()
    return raw_scores >= cutoff
