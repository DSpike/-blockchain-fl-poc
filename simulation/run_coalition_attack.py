"""
Coalition attack evaluation for FairPoC (TIFS Table V / Appendix A).

Simulates coordinated multi-client attacks where m=2 or m=3 clients collude.
Measured against FedAVG, FLTrust, Multi-Krum, FairPoC (ours), and FairPoC-MC (ours_mc).

Attack types:
  coalition_freerider  : all m attackers return the global model unchanged
  coalition_gradscale  : all m attackers scale their update by 10x
  coalition_labelflip  : all m attackers train on fully flipped labels

Metrics per (m, attack, method):
  final_acc      : global model accuracy (%)
  gt_fairness    : Jain's J of reward_i / gt_i
  rounds_detected: first round at which ALL coalition members are slashed (FairPoC only)
  coalition_reward_share: fraction of total reward captured by coalition
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy, csv
import torch
import numpy as np
import torch.nn.functional as F

from fl.client import FLClient
from fl.aggregator import weighted_fedavg
from fl.contribution import (compute_contribution_scores, filter_byzantine,
                              flatten_params, compute_data_quality_gt,
                              evaluate_accuracy, fltrust_scores, krum_select)
from evaluation.metrics import accuracy, fairness_index
from config import (NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS,
                    LOCAL_BATCH_SIZE, LEARNING_RATE, ALPHA_ACC, BETA_DIV,
                    REWARD_PER_ROUND)

SEED           = 42
ALPHA_D        = 0.1       # hard non-IID — worst case for coalition attacks
SCALE_FACTOR   = 10.0
SLASH_THRESHOLD = 0.30

COALITION_SIZES = [2, 3]
ATTACKS         = ["coalition_freerider", "coalition_gradscale", "coalition_labelflip"]
METHODS         = ["fedavg", "fltrust", "krum", "ours", "ours_mc"]


def _build_loaders(alpha, num_clients, batch_size):
    from data.cifar10_loader import build_client_loaders
    return build_client_loaders(alpha=alpha, num_clients=num_clients,
                                batch_size=batch_size, num_workers=0)


def _build_model():
    from models.resnet8 import build_model
    return build_model()


def _inject_coalition(attack_type, coalition_idxs, local_models, global_model,
                      client_loaders, n_classes, device):
    """Modify updates of all coalition members in-place."""
    for idx in coalition_idxs:
        if attack_type == "coalition_freerider":
            local_models[idx].load_state_dict(global_model.state_dict())

        elif attack_type == "coalition_gradscale":
            gvec = flatten_params(global_model)
            with torch.no_grad():
                for p_l, p_g in zip(local_models[idx].parameters(),
                                     global_model.parameters()):
                    delta    = p_l.data - p_g.data
                    p_l.data = p_g.data + delta * SCALE_FACTOR

        elif attack_type == "coalition_labelflip":
            xs, ys = [], []
            for x, y in client_loaders[idx]:
                xs.append(x)
                ys.append(n_classes - 1 - y)
            from torch.utils.data import TensorDataset, DataLoader
            flip_loader = DataLoader(
                TensorDataset(torch.cat(xs), torch.cat(ys)),
                batch_size=client_loaders[idx].batch_size, shuffle=True)
            local_models[idx].load_state_dict(global_model.state_dict())
            opt = torch.optim.SGD(local_models[idx].parameters(),
                                  lr=LEARNING_RATE, momentum=0.9)
            local_models[idx].train()
            for _ in range(LOCAL_EPOCHS):
                for x, y in flip_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    torch.nn.functional.cross_entropy(
                        local_models[idx](x), y).backward()
                    opt.step()


def run_one(method, attack_type, coalition_idxs, cfg,
            client_loaders, val_loader, full_val_loader,
            partitions, targets, device):
    n         = cfg["num_clients"]
    n_classes = len(np.unique(targets))
    data_gt   = compute_data_quality_gt(partitions, targets,
                                        num_classes=n_classes, gamma=0.5)
    clip      = "median" if method == "ours_mc" else "max"

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    global_model = _build_model().to(device)
    clients = [
        FLClient(i, global_model, client_loaders[i],
                 lr=cfg["lr"], local_epochs=cfg["local_epochs"], device=device)
        for i in range(n)]

    cumulative_rewards  = np.zeros(n)
    cumulative_slashed  = np.zeros(n)
    blacklisted         = np.zeros(n, dtype=bool)
    slash_threshold_val = SLASH_THRESHOLD / n
    rounds_all_detected = None   # first round all coalition members slashed

    for rnd in range(1, cfg["num_rounds"] + 1):
        for c in clients:
            c.set_global_weights(global_model.state_dict())
        local_models = [c.train() for c in clients]

        _inject_coalition(attack_type, coalition_idxs, local_models,
                          global_model, client_loaders, n_classes, device)

        scores, raw = compute_contribution_scores(
            local_models, global_model, val_loader, device,
            alpha=cfg["alpha_poc"], beta=cfg["beta_poc"], grad_norm_clip=clip)

        keep = np.ones(n, dtype=bool)

        if method == "fedavg":
            weights = np.ones(n) / n
            weighted_fedavg(global_model, local_models, weights)

        elif method == "fltrust":
            ts, _   = fltrust_scores(local_models, global_model,
                                     val_loader, device,
                                     lr=cfg["lr"], local_epochs=1)
            keep    = ts > 1e-6
            if not keep.any():
                keep = np.ones(n, dtype=bool)
            active  = [m for m, k in zip(local_models, keep) if k]
            weights = np.zeros(n)
            weights[keep] = ts[keep] / ts[keep].sum()
            weighted_fedavg(global_model, active, ts[keep] / ts[keep].sum())

        elif method == "krum":
            f_byz   = len(coalition_idxs)
            keep    = krum_select(local_models, global_model,
                                  n_byzantine=f_byz)
            active  = [m for m, k in zip(local_models, keep) if k]
            w_uni   = np.ones(len(active)) / len(active)
            weights = np.zeros(n)
            weights[keep] = w_uni
            weighted_fedavg(global_model, active, w_uni)

        elif method in ("ours", "ours_mc"):
            slash_pool = 0.0
            for i in range(n):
                if scores[i] < slash_threshold_val and not blacklisted[i]:
                    cumulative_slashed[i] += 100
                    slash_pool            += 100
                    blacklisted[i]         = True
                    keep[i]                = False

            if rounds_all_detected is None:
                if all(blacklisted[idx] for idx in coalition_idxs):
                    rounds_all_detected = rnd

            active_mask = keep & ~blacklisted
            if not active_mask.any():
                active_mask = np.ones(n, dtype=bool) & ~blacklisted
            if not active_mask.any():
                active_mask = np.ones(n, dtype=bool)

            active  = [m for m, k in zip(local_models, active_mask) if k]
            w       = scores[active_mask]
            weights = np.zeros(n)
            weights[active_mask] = w / (w.sum() + 1e-9)
            weighted_fedavg(global_model, active, w / (w.sum() + 1e-9))

            total_pool = REWARD_PER_ROUND + slash_pool
            honest_mask = active_mask
            if honest_mask.any():
                hs = scores[honest_mask]
                cumulative_rewards[honest_mask] += (hs / hs.sum()) * total_pool
            continue   # reward already accumulated

        # for non-ours methods
        cumulative_rewards += weights * REWARD_PER_ROUND

    final_acc = accuracy(global_model, full_val_loader, device)
    r_norm    = cumulative_rewards / (cumulative_rewards.sum() + 1e-12) + 1e-12
    fair      = fairness_index(data_gt, r_norm)

    coalition_share = cumulative_rewards[coalition_idxs].sum() / \
                      (cumulative_rewards.sum() + 1e-9)
    fair_share      = len(coalition_idxs) / n

    return {
        "final_acc":            final_acc * 100,
        "gt_fairness":          fair,
        "coalition_share":      coalition_share,
        "expected_share":       fair_share,
        "overreward":           coalition_share / fair_share,
        "rounds_all_detected":  rounds_all_detected or cfg["num_rounds"],
        "any_slashed":          int(rounds_all_detected is not None),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    cfg = {
        "num_clients":      NUM_CLIENTS,
        "num_rounds":       NUM_ROUNDS,
        "local_epochs":     LOCAL_EPOCHS,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "lr":               LEARNING_RATE,
        "alpha_poc":        ALPHA_ACC,
        "beta_poc":         BETA_DIV,
        "seed":             SEED,
        "alpha_dirichlet":  ALPHA_D,
    }

    client_loaders, val_loader, partitions, full_val_loader, targets = \
        _build_loaders(ALPHA_D, NUM_CLIENTS, LOCAL_BATCH_SIZE)

    rows = []
    for m_size in COALITION_SIZES:
        coalition_idxs = list(range(m_size))   # clients 0..m-1 collude
        for attack in ATTACKS:
            print(f"\n{'='*65}")
            print(f"  Coalition size m={m_size}  |  {attack}")
            print(f"{'='*65}")
            for method in METHODS:
                r = run_one(method, attack, coalition_idxs, cfg,
                            client_loaders, val_loader, full_val_loader,
                            partitions, targets, device)
                rows.append({
                    "coalition_size": m_size,
                    "attack":         attack,
                    "method":         method,
                    **r,
                })
                print(f"  {method:10s}  acc={r['final_acc']:.2f}%  "
                      f"fair={r['gt_fairness']:.4f}  "
                      f"over={r['overreward']:.2f}x  "
                      f"detected_round={r['rounds_all_detected']}", flush=True)

    os.makedirs("results", exist_ok=True)
    out = "results/coalition_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
