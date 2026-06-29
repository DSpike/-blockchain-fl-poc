"""
PoCL baseline: Proof-of-Collaborative-Learning (arxiv:2407.13018).
IID-only contribution scoring — uses validation accuracy alone, no grad divergence.
This is our main competitor.
"""

import numpy as np
from fl.contribution import evaluate_accuracy


def pocl_scores(client_models, val_loader, device):
    """Score = validation accuracy only (no heterogeneity term)."""
    accs = np.array([evaluate_accuracy(m, val_loader, device) for m in client_models])
    return accs / (accs.sum() + 1e-9)
