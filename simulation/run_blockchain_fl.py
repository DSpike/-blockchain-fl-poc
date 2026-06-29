"""
Full system: FL + blockchain layer.
Requires Ganache running and contracts deployed (python blockchain/deploy.py).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json, hashlib, copy
import torch, numpy as np
import argparse, yaml

from config import *
from models.resnet8 import build_model
from data.cifar10_loader import build_client_loaders
from fl.client import FLClient
from fl.aggregator import weighted_fedavg
from fl.contribution import compute_contribution_scores, filter_byzantine
from evaluation.metrics import accuracy, fairness_index, ConvergenceTracker
from blockchain.web3_interface import connect, load_contract, submit_contribution, distribute_rewards


def model_hash(model):
    params = b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    return hashlib.sha256(params).hexdigest()


def run(cfg, contracts_path="deployed_contracts.json"):
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # blockchain setup
    w3 = connect(cfg.get("ganache_url", GANACHE_URL))
    accounts = w3.eth.accounts
    with open(contracts_path) as f:
        deployed = json.load(f)

    reg_contract = load_contract(w3, None, deployed["ContributionRegistry"]["address"])
    rew_contract = load_contract(w3, None, deployed["RewardDistributor"]["address"])

    client_loaders, val_loader, _ = build_client_loaders(
        alpha=cfg["alpha_dirichlet"], num_clients=cfg["num_clients"],
        batch_size=cfg["local_batch_size"])

    global_model = build_model().to(device)
    clients = [
        FLClient(i, global_model, client_loaders[i],
                 lr=cfg["lr"], local_epochs=cfg["local_epochs"], device=device)
        for i in range(cfg["num_clients"])
    ]

    tracker = ConvergenceTracker()

    for rnd in range(1, cfg["num_rounds"] + 1):
        for c in clients:
            c.set_global_weights(global_model.state_dict())
            c.train()

        local_models = [c.model for c in clients]
        scores, raw = compute_contribution_scores(
            local_models, global_model, val_loader, device,
            alpha=cfg["alpha_poc"], beta=cfg["beta_poc"])

        keep_mask = filter_byzantine(scores, raw)
        active_models = [m for m, k in zip(local_models, keep_mask) if k]
        active_scores = scores[keep_mask]
        active_scores /= active_scores.sum()

        # --- on-chain submission ---
        for idx, (keep, score) in enumerate(zip(keep_mask, scores)):
            if keep:
                mhash = model_hash(local_models[idx])
                submit_contribution(reg_contract, accounts[idx + 1],
                                    rnd, mhash, score, accounts[0])

        weighted_fedavg(global_model, active_models, active_scores)

        # --- on-chain reward distribution ---
        active_addrs = [accounts[i + 1] for i, k in enumerate(keep_mask) if k]
        distribute_rewards(rew_contract, active_addrs,
                           active_scores.tolist(), REWARD_PER_ROUND, rnd, accounts[0])

        acc = accuracy(global_model, val_loader, device)
        fair = fairness_index(raw["raw"][keep_mask], active_scores)
        tracker.log(rnd, acc, fair)

        if rnd % 10 == 0:
            print(f"Round {rnd:3d} | Acc: {acc:.4f} | Fairness: {fair:.4f}")

    return tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    # fill defaults
    cfg.setdefault("num_clients", NUM_CLIENTS)
    cfg.setdefault("num_rounds", NUM_ROUNDS)
    cfg.setdefault("local_epochs", LOCAL_EPOCHS)
    cfg.setdefault("local_batch_size", LOCAL_BATCH_SIZE)
    cfg.setdefault("lr", LEARNING_RATE)
    cfg.setdefault("alpha_dirichlet", DIRICHLET_ALPHA)
    cfg.setdefault("alpha_poc", ALPHA_ACC)
    cfg.setdefault("beta_poc", BETA_DIV)
    cfg.setdefault("seed", SEED)
    run(cfg)
