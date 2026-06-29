"""Evaluation metrics: accuracy, fairness index, convergence tracking."""

import numpy as np


def accuracy(model, loader, device):
    import torch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def fairness_index(contributions, rewards):
    """
    Jain's fairness index on the ratio reward_i / contribution_i.
    Score of 1.0 = perfectly fair; lower = more unfair.
    """
    ratios = rewards / (contributions + 1e-9)
    n = len(ratios)
    return (ratios.sum() ** 2) / (n * (ratios ** 2).sum() + 1e-9)


def contribution_reward_correlation(contributions, rewards):
    """Pearson correlation between contribution scores and reward amounts."""
    return float(np.corrcoef(contributions, rewards)[0, 1])


class ConvergenceTracker:
    def __init__(self):
        self.rounds = []
        self.accuracies = []
        self.fairness = []

    def log(self, rnd, acc, fair):
        self.rounds.append(rnd)
        self.accuracies.append(acc)
        self.fairness.append(fair)

    def to_dict(self):
        return {"round": self.rounds, "accuracy": self.accuracies, "fairness": self.fairness}
