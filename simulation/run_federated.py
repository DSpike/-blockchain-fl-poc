"""
FL-only training loop (no blockchain overhead).
Use this to validate the PoC mechanism quickly before adding the blockchain layer.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import torch
import numpy as np
import yaml
import argparse

from config import *
from models.resnet8 import build_model
from data.cifar10_loader import build_client_loaders
from fl.client import FLClient
from fl.aggregator import weighted_fedavg
from fl.contribution import compute_contribution_scores, filter_byzantine
from evaluation.metrics import accuracy, fairness_index, ConvergenceTracker
from evaluation.byzantine import inject_byzantine


def load_config(yaml_path=None):
    cfg = {
        "num_clients": NUM_CLIENTS, "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS, "local_batch_size": LOCAL_BATCH_SIZE,
        "lr": LEARNING_RATE, "alpha_dirichlet": DIRICHLET_ALPHA,
        "alpha_poc": ALPHA_ACC, "beta_poc": BETA_DIV,
        "byzantine_fraction": BYZANTINE_FRACTION,
        "byzantine_attack": BYZANTINE_ATTACK, "seed": SEED,
    }
    if yaml_path:
        with open(yaml_path) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def run(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Clients: {cfg['num_clients']} | α={cfg['alpha_dirichlet']}")

    client_loaders, val_loader, _, full_val_loader = build_client_loaders(
        alpha=cfg["alpha_dirichlet"],
        num_clients=cfg["num_clients"],
        batch_size=cfg["local_batch_size"],
    )

    global_model = build_model().to(device)
    clients = [
        FLClient(i, global_model, client_loaders[i],
                 lr=cfg["lr"], local_epochs=cfg["local_epochs"], device=device)
        for i in range(cfg["num_clients"])
    ]

    tracker = ConvergenceTracker()
    reward_pool = np.zeros(cfg["num_clients"])

    for rnd in range(1, cfg["num_rounds"] + 1):
        # --- local training ---
        for c in clients:
            c.set_global_weights(global_model.state_dict())
            c.train()

        local_models = [c.model for c in clients]

        # --- optional Byzantine injection ---
        if cfg["byzantine_fraction"] > 0:
            local_models, _ = inject_byzantine(
                local_models, cfg["byzantine_fraction"], cfg["byzantine_attack"])

        # --- contribution scoring (THE NOVELTY) ---
        scores, raw = compute_contribution_scores(
            local_models, global_model, val_loader, device,
            alpha=cfg["alpha_poc"], beta=cfg["beta_poc"])

        keep_mask = filter_byzantine(scores, raw)
        active_models = [m for m, k in zip(local_models, keep_mask) if k]
        active_scores = scores[keep_mask]
        active_scores /= active_scores.sum()

        # --- weighted aggregation ---
        weighted_fedavg(global_model, active_models, active_scores)

        # --- reward distribution ---
        reward_pool[keep_mask] += active_scores * REWARD_PER_ROUND

        # --- evaluate ---
        acc = accuracy(global_model, full_val_loader, device)
        fair = fairness_index(raw["raw"][keep_mask], active_scores)
        tracker.log(rnd, acc, fair)

        if rnd % 10 == 0:
            print(f"Round {rnd:3d} | Acc: {acc:.4f} | Fairness: {fair:.4f}", flush=True)

    return tracker, reward_pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)
