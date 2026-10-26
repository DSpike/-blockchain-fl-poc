"""
Adversarial robustness evaluation for FairPoC.

Attacks (one client out of 10):
  GradScale  : attacker scales its update by 10x (gradient inflation)
  FreeRider  : attacker returns the global model unchanged (zero update)
  LabelFlip  : attacker trains on labels flipped to (C-1 - y)

Methods evaluated:
  fedavg      : uniform aggregation (no scoring)
  blade_fl    : cosine-similarity filter (BLADE-FL baseline)
  ours        : FairPoC with max-normalisation (default)
  ours_mc     : FairPoC with median-clip normalisation (GradScale defence)

Metrics per (attack, method):
  final_acc       : final-round global model accuracy
  gt_fairness     : GT-Fairness (Jain's J of reward / gt_i)
  overreward      : attacker_reward_share / (1/n_clients)
                    1.0 = fair; >1 = over-rewarded; <1 = under-rewarded

Usage:
    cd blockchain-fl-poc
    python simulation/run_adversarial.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy, csv, time
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
ALPHA_D        = 0.1
SCALE_FACTOR   = 10.0
ATTACKER_IDX   = 0
METHODS        = ["fedavg", "blade_fl", "fltrust", "krum", "ours", "ours_mc"]

# Stake-and-slash parameters (mirrors FairPoC.sol constants)
STAKE_AMOUNT       = 1000   # tokens staked per client per round
SLASH_THRESHOLD    = 0.30   # slash if score < 30% of equal share (1/n)


def _build_loaders(dataset, alpha, num_clients, batch_size):
    if dataset == "cifar10":
        from data.cifar10_loader import build_client_loaders
    else:
        from data.fashionmnist_loader import build_client_loaders
    return build_client_loaders(alpha=alpha, num_clients=num_clients,
                                batch_size=batch_size)


def _build_model(dataset="cifar10"):
    if dataset == "cifar10":
        from models.resnet8 import build_model
    else:
        from models.simplecnn import build_model
    return build_model()


def _inject_attack(attack_type, local_models, global_model, client_loaders,
                   n_classes, device):
    """Modify local_models[ATTACKER_IDX] in-place for the given attack."""
    if attack_type == "none":
        return  # no-op: clean baseline

    elif attack_type == "gradscale":
        gvec = flatten_params(global_model)
        with torch.no_grad():
            for p_l, p_g in zip(local_models[ATTACKER_IDX].parameters(),
                                 global_model.parameters()):
                delta = p_l.data - p_g.data
                p_l.data = p_g.data + delta * SCALE_FACTOR

    elif attack_type == "freerider":
        local_models[ATTACKER_IDX].load_state_dict(global_model.state_dict())

    elif attack_type == "labelflip":
        # Re-train attacker with flipped labels on its own loader
        atk_loader = _make_flip_loader(client_loaders[ATTACKER_IDX],
                                        n_classes, device)
        local_models[ATTACKER_IDX].load_state_dict(global_model.state_dict())
        opt = torch.optim.SGD(local_models[ATTACKER_IDX].parameters(),
                               lr=LEARNING_RATE, momentum=0.9)
        local_models[ATTACKER_IDX].train()
        for _ in range(LOCAL_EPOCHS):
            for x, y in atk_loader:
                x = x.to(device)
                y_flip = (n_classes - 1 - y).to(device)
                opt.zero_grad()
                torch.nn.functional.cross_entropy(
                    local_models[ATTACKER_IDX](x), y_flip).backward()
                opt.step()


def _make_flip_loader(loader, n_classes, device):
    """Return a DataLoader with flipped labels."""
    from torch.utils.data import TensorDataset, DataLoader
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(n_classes - 1 - y)
    X = torch.cat(xs)
    Y = torch.cat(ys)
    return DataLoader(TensorDataset(X, Y), batch_size=loader.batch_size,
                      shuffle=True)


def run_one_method(method, attack_type, cfg, client_loaders, val_loader,
                   full_val_loader, partitions, targets, device, dataset="cifar10"):
    n          = cfg["num_clients"]
    n_classes  = len(np.unique(targets))
    data_gt    = compute_data_quality_gt(partitions, targets,
                                         num_classes=n_classes, gamma=0.5)
    clip       = "median" if method == "ours_mc" else "max"
    cfg        = dict(cfg)   # local copy so we can add lr safely

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    global_model = _build_model(dataset).to(device)
    clients = [
        FLClient(i, global_model, client_loaders[i],
                 lr=cfg["lr"], local_epochs=cfg["local_epochs"],
                 device=device)
        for i in range(n)]

    cumulative_rewards = np.zeros(n)
    cumulative_slashed = np.zeros(n)   # stake lost to slash per client
    slash_threshold    = SLASH_THRESHOLD / n  # score fraction below which slash fires
    blacklisted        = np.zeros(n, dtype=bool)

    for rnd in range(1, cfg["num_rounds"] + 1):
        for c in clients:
            c.set_global_weights(global_model.state_dict())
        local_models = [c.train() for c in clients]
        _inject_attack(attack_type, local_models, global_model,
                       client_loaders, n_classes, device)

        scores, raw = compute_contribution_scores(
            local_models, global_model, val_loader, device,
            alpha=cfg["alpha_poc"], beta=cfg["beta_poc"],
            grad_norm_clip=clip)
        keep = filter_byzantine(scores, raw)

        # --- stake-and-slash (FairPoC / FairPoC-MC only) ---
        slash_pool = 0.0
        if method in ("ours", "ours_mc"):
            slashed_this_round = np.zeros(n, dtype=bool)
            for i in range(n):
                if scores[i] < slash_threshold and not blacklisted[i]:
                    cumulative_slashed[i] += STAKE_AMOUNT
                    slash_pool            += STAKE_AMOUNT
                    blacklisted[i]         = True   # permanent exclusion
                    keep[i]                = False
                    slashed_this_round[i]  = True

        if method == "fedavg":
            keep    = np.ones(n, dtype=bool)
            weights = np.ones(n) / n
            weighted_fedavg(global_model, local_models, weights)

        elif method == "blade_fl":
            gvec   = flatten_params(global_model)
            grads  = [flatten_params(m) - gvec for m in local_models]
            mean_g = torch.stack(grads).mean(dim=0)
            cs     = np.array([F.cosine_similarity(g.unsqueeze(0),
                               mean_g.unsqueeze(0)).item() for g in grads])
            cutoff = cs.mean() - 2.0 * cs.std()
            keep   = cs >= cutoff
            active = [m for m, k in zip(local_models, keep) if k]
            if not active:
                active = local_models
                keep   = np.ones(n, dtype=bool)
            weights = np.ones(len(active)) / len(active)
            weighted_fedavg(global_model, active, weights)

        elif method == "fltrust":
            ts, _   = fltrust_scores(local_models, global_model, val_loader, device,
                                     lr=cfg["lr"], local_epochs=1)
            keep    = ts > 1e-6
            if not keep.any():
                keep = np.ones(n, dtype=bool)
            active  = [m for m, k in zip(local_models, keep) if k]
            weights = ts[keep] / ts[keep].sum()
            weighted_fedavg(global_model, active, weights)

        elif method == "krum":
            f_byz   = max(1, int(round(0.2 * n)))
            keep    = krum_select(local_models, global_model, n_byzantine=f_byz)
            active  = [m for m, k in zip(local_models, keep) if k]
            weights = np.ones(len(active)) / len(active)
            weighted_fedavg(global_model, active, weights)

        elif method in ("ours", "ours_mc"):
            # exclude blacklisted clients from aggregation
            active_mask = keep & ~blacklisted
            if not active_mask.any():
                active_mask = ~blacklisted
            active  = [m for m, k in zip(local_models, active_mask) if k]
            w       = scores[active_mask]
            weights = w / (w.sum() + 1e-9)
            weighted_fedavg(global_model, active, weights)

        full_reward = np.zeros(n)
        if method in ("ours", "ours_mc"):
            honest_mask = keep & ~blacklisted
            if honest_mask.any():
                honest_scores = scores[honest_mask]
                honest_weights = honest_scores / honest_scores.sum()
                # slash pool redistributed to honest clients this round
                total_pool = REWARD_PER_ROUND + slash_pool
                full_reward[honest_mask] = honest_weights * total_pool
        else:
            full_reward[keep] = weights * REWARD_PER_ROUND

        cumulative_rewards += full_reward

    final_acc = accuracy(global_model, full_val_loader, device)
    fair = fairness_index(data_gt,
                          cumulative_rewards / (cumulative_rewards.sum() + 1e-12)
                          + 1e-12)
    total_r        = cumulative_rewards.sum()
    attacker_share = cumulative_rewards[ATTACKER_IDX] / total_r if total_r else 0.0
    expected_share = 1.0 / n
    # net gain = rewards earned - stake lost (economic deterrent measure)
    net_gain_ratio = (cumulative_rewards[ATTACKER_IDX] - cumulative_slashed[ATTACKER_IDX]) \
                     / (expected_share * total_r + 1e-9)

    return {
        "final_acc":       final_acc,
        "gt_fairness":     fair,
        "attacker_share":  attacker_share,
        "overreward":      attacker_share / expected_share,
        "stake_slashed":   cumulative_slashed[ATTACKER_IDX],
        "net_gain_ratio":  net_gain_ratio,   # <1 = net loss for attacker
    }


def run_dataset(dataset, cfg, device, attacks):
    """Run all attacks for one dataset. Returns dict keyed by attack→method."""
    print(f"\n{'#'*65}")
    print(f"  DATASET: {dataset.upper()}")
    print(f"{'#'*65}")

    client_loaders, val_loader, partitions, full_val_loader, targets = \
        _build_loaders(dataset, cfg["alpha_dirichlet"],
                       cfg["num_clients"], cfg["local_batch_size"])

    results = {}
    for attack in attacks:
        print(f"\n{'='*65}")
        print(f"  ATTACK: {attack.upper()}")
        print(f"{'='*65}")
        results[attack] = {}
        for method in METHODS:
            r = run_one_method(method, attack, cfg, client_loaders, val_loader,
                               full_val_loader, partitions, targets, device,
                               dataset=dataset)
            results[attack][method] = r
            print(f"  {method:10s}  acc={r['final_acc']*100:.2f}%  "
                  f"fair={r['gt_fairness']:.4f}  "
                  f"overreward={r['overreward']:.2f}x  "
                  f"net_gain={r.get('net_gain_ratio', float('nan')):.2f}x")

    # clean baseline
    print(f"\n{'='*65}")
    print(f"  BASELINE: No attack")
    print(f"{'='*65}")
    results["clean"] = {}
    for method in ("ours", "ours_mc"):
        r = run_one_method(method, "none", cfg, client_loaders, val_loader,
                           full_val_loader, partitions, targets, device,
                           dataset=dataset)
        results["clean"][method] = r
        print(f"  {method:10s}  acc={r['final_acc']*100:.2f}%  "
              f"fair={r['gt_fairness']:.4f}")
    return results


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    attacks  = ["gradscale", "freerider", "labelflip"]
    datasets = ["cifar10", "fashionmnist"]
    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds, cfg, device, attacks)

    # save combined CSV
    os.makedirs("results", exist_ok=True)
    out_path = "results/adversarial_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "attack", "method", "final_acc_%",
                         "gt_fairness", "attacker_share", "expected_share",
                         "overreward", "stake_slashed", "net_gain_ratio"])
        for ds in datasets:
            for attack in attacks + ["clean"]:
                methods_in = METHODS if attack != "clean" else ["ours", "ours_mc"]
                for method in methods_in:
                    v = all_results[ds][attack][method]
                    writer.writerow([ds, attack, method,
                                     v["final_acc"]*100, v["gt_fairness"],
                                     v["attacker_share"], 1.0/NUM_CLIENTS,
                                     v["overreward"],
                                     v.get("stake_slashed", 0),
                                     v.get("net_gain_ratio", "N/A")])

    print(f"\nSaved: {out_path}")
    print("\n--- Summary ---")
    print(f"{'DS':<12} {'Attack':<12} {'Method':<10} {'Acc%':>7} "
          f"{'GT-Fair':>8} {'Overreward':>11} {'NetGain':>9}")
    print("-" * 75)
    for ds in datasets:
        for attack in attacks:
            for method in METHODS:
                v = all_results[ds][attack][method]
                ng = v.get("net_gain_ratio", float("nan"))
                ng_str = f"{ng:>8.2f}x" if isinstance(ng, float) else "     N/A"
                print(f"  {ds:<11} {attack:<11} {method:<10} "
                      f"{v['final_acc']*100:>6.2f}  "
                      f"{v['gt_fairness']:>7.4f}  "
                      f"{v['overreward']:>10.2f}x  {ng_str}")


if __name__ == "__main__":
    main()
