"""Utilities for inspecting and visualising data partitions."""

import numpy as np
import matplotlib.pyplot as plt


def class_distribution(partitions, targets, num_classes=10):
    """Returns (num_clients x num_classes) matrix of class counts."""
    targets = np.array(targets)
    dist = np.zeros((len(partitions), num_classes), dtype=int)
    for i, idxs in enumerate(partitions):
        for c in range(num_classes):
            dist[i, c] = np.sum(targets[idxs] == c)
    return dist


def heterogeneity_score(dist):
    """
    Earth Mover's Distance-based heterogeneity: mean JS divergence of each
    client's distribution from the uniform distribution.
    Higher = more heterogeneous.
    """
    uniform = np.ones(dist.shape[1]) / dist.shape[1]
    scores = []
    for row in dist:
        p = row / row.sum() if row.sum() > 0 else uniform
        m = 0.5 * (p + uniform)
        jsd = 0.5 * np.sum(p * np.log((p + 1e-9) / (m + 1e-9))) + \
              0.5 * np.sum(uniform * np.log((uniform + 1e-9) / (m + 1e-9)))
        scores.append(jsd)
    return float(np.mean(scores))


def plot_distribution(dist, alpha, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(dist.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, dist.shape[1]))
    for c in range(dist.shape[1]):
        ax.bar(range(dist.shape[0]), dist[:, c], bottom=bottom,
               color=colors[c], label=f"Class {c}")
        bottom += dist[:, c]
    ax.set_xlabel("Client")
    ax.set_ylabel("Samples")
    ax.set_title(f"Class distribution across clients (Dirichlet α={alpha})")
    ax.legend(loc="upper right", ncol=5, fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
