"""
Full system: FL + blockchain layer.
Requires Ganache running and contracts deployed (python blockchain/deploy.py).

Measures and logs:
  - Per-round gas cost (ContributionRegistry submissions + RewardDistributor)
  - Transaction latency (seconds per round for on-chain ops)
  - Accuracy, fairness (same as FL-only run)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json, hashlib, time, csv
import torch, numpy as np
import argparse, yaml

from config import *
from models.resnet8 import build_model
from data.cifar10_loader import build_client_loaders
from fl.client import FLClient
from fl.aggregator import weighted_fedavg
from fl.contribution import compute_contribution_scores, filter_byzantine
from evaluation.metrics import accuracy, fairness_index, ConvergenceTracker
from blockchain.web3_interface import connect, submit_contribution, distribute_rewards


def load_contracts(w3, contracts_path="deployed_contracts.json"):
    with open(contracts_path) as f:
        deployed = json.load(f)
    contracts = {}
    for name, info in deployed.items():
        contracts[name] = w3.eth.contract(
            address=w3.to_checksum_address(info["address"]),
            abi=info["abi"]
        )
    return contracts


def model_hash(model):
    params = b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    return hashlib.sha256(params).hexdigest()


def run(cfg, contracts_path="deployed_contracts.json", save_csv=True):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── blockchain setup ─────────────────────────────────────────────────────
    w3 = connect(cfg.get("ganache_url", GANACHE_URL))
    accounts = w3.eth.accounts
    contracts = load_contracts(w3, contracts_path)
    reg = contracts["ContributionRegistry"]
    rew = contracts["RewardDistributor"]
    owner = accounts[0]

    print(f"Connected to Ganache | Accounts: {len(accounts)}", flush=True)
    print(f"Device: {device} | α={cfg['alpha_dirichlet']} | Rounds: {cfg['num_rounds']}", flush=True)
    print("=" * 65, flush=True)

    # ── data + model ─────────────────────────────────────────────────────────
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
    gas_log = []      # (round, total_gas, latency_sec)
    reward_log = []   # (round, client_idx, reward_weight)

    for rnd in range(1, cfg["num_rounds"] + 1):
        # ── local training ───────────────────────────────────────────────────
        for c in clients:
            c.set_global_weights(global_model.state_dict())
            c.train()

        local_models = [c.model for c in clients]

        # ── contribution scoring ─────────────────────────────────────────────
        scores, raw = compute_contribution_scores(
            local_models, global_model, val_loader, device,
            alpha=cfg["alpha_poc"], beta=cfg["beta_poc"])

        keep_mask = filter_byzantine(scores, raw)
        active_models  = [m for m, k in zip(local_models, keep_mask) if k]
        active_scores  = scores[keep_mask]
        active_scores /= active_scores.sum()

        # ── on-chain submission ──────────────────────────────────────────────
        t0 = time.time()
        total_gas = 0

        for idx, (keep, score) in enumerate(zip(keep_mask, scores)):
            if keep:
                mhash = model_hash(local_models[idx])
                tx_hash = submit_contribution(
                    reg, accounts[idx + 1], rnd, mhash, float(score), owner)
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
                total_gas += receipt.gasUsed

        # ── weighted aggregation ─────────────────────────────────────────────
        weighted_fedavg(global_model, active_models, active_scores)

        # ── on-chain reward distribution ─────────────────────────────────────
        active_addrs = [accounts[i + 1] for i, k in enumerate(keep_mask) if k]
        tx_hash = distribute_rewards(
            rew, active_addrs, active_scores.tolist(),
            REWARD_PER_ROUND, rnd, owner)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        total_gas += receipt.gasUsed

        latency = time.time() - t0
        gas_log.append((rnd, total_gas, latency))

        # log per-client rewards
        for i, (keep, w) in enumerate(zip(keep_mask, scores)):
            if keep:
                reward_log.append((rnd, i, float(w)))

        # ── evaluate ─────────────────────────────────────────────────────────
        acc  = accuracy(global_model, full_val_loader, device)
        fair = fairness_index(raw["raw"][keep_mask], active_scores)
        tracker.log(rnd, acc, fair)

        if rnd % 10 == 0:
            avg_gas = sum(g for _, g, _ in gas_log[-10:]) / 10
            avg_lat = sum(l for _, _, l in gas_log[-10:]) / 10
            print(f"  Round {rnd:3d} | Acc: {acc:.4f} | Fairness: {fair:.4f} | "
                  f"Gas: {avg_gas:.0f} | Latency: {avg_lat:.2f}s", flush=True)

    # ── summary ──────────────────────────────────────────────────────────────
    total_gas_all = sum(g for _, g, _ in gas_log)
    avg_latency   = sum(l for _, _, l in gas_log) / len(gas_log)
    print(f"\n{'='*65}", flush=True)
    print(f"  Total gas used:      {total_gas_all:,}", flush=True)
    print(f"  Avg gas per round:   {total_gas_all // len(gas_log):,}", flush=True)
    print(f"  Avg latency/round:   {avg_latency:.3f}s", flush=True)
    print(f"  Final accuracy:      {tracker.accuracies[-1]:.4f}", flush=True)
    print(f"  Avg fairness:        {np.mean(tracker.fairness):.4f}", flush=True)
    print(f"{'='*65}\n", flush=True)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    if save_csv:
        os.makedirs("results", exist_ok=True)

        with open("results/blockchain_metrics.csv", "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["round", "total_gas", "latency_sec"])
            w_.writerows(gas_log)
        print("  Saved: results/blockchain_metrics.csv", flush=True)

        with open("results/blockchain_convergence.csv", "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["round", "accuracy", "fairness"])
            for i, r in enumerate(tracker.rounds):
                w_.writerow([r, tracker.accuracies[i], tracker.fairness[i]])
        print("  Saved: results/blockchain_convergence.csv", flush=True)

    return tracker, gas_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--contracts", type=str, default="deployed_contracts.json")
    args = parser.parse_args()

    cfg = {
        "num_clients": NUM_CLIENTS, "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS, "local_batch_size": LOCAL_BATCH_SIZE,
        "lr": LEARNING_RATE, "alpha_dirichlet": DIRICHLET_ALPHA,
        "alpha_poc": ALPHA_ACC, "beta_poc": BETA_DIV, "seed": SEED,
    }
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    run(cfg, contracts_path=args.contracts)
