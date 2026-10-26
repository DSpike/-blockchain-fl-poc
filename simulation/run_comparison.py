"""
Comparison runner: FedAVG, ValScore, CosineFilter, FairPoC.

Fairness is measured as Jain's index of reward_i / data_quality_gt_i,
where data_quality_gt is an INDEPENDENT ground truth derived purely from
the client data partitions (volume fraction + normalised class entropy).
This is not circular: gt does not depend on any method's score.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy, csv, argparse, yaml
import torch
import numpy as np

from config import *


def _get_model_and_loader(dataset):
    if dataset == "fashionmnist":
        from models.simplecnn import build_model
        from data.fashionmnist_loader import build_client_loaders
    elif dataset == "cifar100":
        from models.resnet8 import build_model as _build
        from data.cifar100_loader import build_client_loaders
        def build_model():
            return _build(num_classes=100)
        return build_model, build_client_loaders
    elif dataset == "svhn":
        from models.resnet8 import build_model
        from data.svhn_loader import build_client_loaders
        return build_model, build_client_loaders
    else:
        from models.resnet8 import build_model
        from data.cifar10_loader import build_client_loaders
    return build_model, build_client_loaders


from fl.client import FLClient
from fl.aggregator import weighted_fedavg
from fl.contribution import (compute_contribution_scores, filter_byzantine,
                              evaluate_accuracy, flatten_params,
                              compute_loo_contributions,
                              compute_data_quality_gt, tmc_shapley,
                              fltrust_scores, krum_select)
from evaluation.metrics import accuracy, fairness_index, ConvergenceTracker
from evaluation.byzantine import inject_byzantine


# ---------------------------------------------------------------------------
# aggregation strategies
# ---------------------------------------------------------------------------

def run_method(method, clients, global_model, val_loader,
               full_val_loader, device, cfg, rng_seed):
    """
    Run one FL round for the given method.

    Returns (global_model, weights, true_poc_scores, raw, keep)
      weights        — per-kept-client reward (normalised, sums to 1)
      true_poc_scores — PoC scores for kept clients (for reference only)
      raw            — dict with val_acc, grad_div per client
      keep           — boolean mask (length = n_clients) indicating which
                       clients are included this round
    """
    n = len(clients)

    for c in clients:
        c.set_global_weights(global_model.state_dict())
        c.train()

    local_models = [c.model for c in clients]

    if cfg["byzantine_fraction"] > 0:
        local_models, _ = inject_byzantine(
            local_models, cfg["byzantine_fraction"],
            cfg["byzantine_attack"], seed=rng_seed,
            global_model=global_model)

    # compute PoC scores (used for FairPoC aggregation and reference logging)
    true_scores, raw = compute_contribution_scores(
        local_models, global_model, val_loader, device,
        alpha=cfg["alpha_poc"], beta=cfg["beta_poc"])

    if method == "fedavg":
        keep    = np.ones(n, dtype=bool)
        weights = np.ones(n) / n
        weighted_fedavg(global_model, local_models, weights)

    elif method == "valscore":
        val_accs = np.array([evaluate_accuracy(m, val_loader, device)
                             for m in local_models])
        cutoff   = val_accs.mean() - 2.0 * val_accs.std()
        keep     = val_accs >= cutoff
        active   = [m for m, k in zip(local_models, keep) if k]
        w        = val_accs[keep]
        weights  = w / w.sum()
        weighted_fedavg(global_model, active, weights)

    elif method == "cosinefilter":
        import torch.nn.functional as F
        global_vec = flatten_params(global_model)
        grads      = [flatten_params(m) - global_vec for m in local_models]
        mean_grad  = torch.stack(grads).mean(dim=0)
        cos_sims   = np.array([
            F.cosine_similarity(g.unsqueeze(0), mean_grad.unsqueeze(0)).item()
            for g in grads])
        cutoff  = cos_sims.mean() - 2.0 * cos_sims.std()
        keep    = cos_sims >= cutoff
        active  = [m for m, k in zip(local_models, keep) if k]
        weights = np.ones(len(active)) / len(active)
        weighted_fedavg(global_model, active, weights)

    elif method == "ours":
        keep    = filter_byzantine(true_scores, raw)
        active  = [m for m, k in zip(local_models, keep) if k]
        w       = true_scores[keep]
        weights = w / w.sum()
        weighted_fedavg(global_model, active, weights)

    elif method == "ours_ip":
        ip_scores, ip_raw = compute_contribution_scores(
            local_models, global_model, val_loader, device,
            alpha=cfg["alpha_poc"], beta=cfg["beta_poc"],
            grad_signal="inner_product")
        keep    = filter_byzantine(ip_scores, ip_raw)
        active  = [m for m, k in zip(local_models, keep) if k]
        w       = ip_scores[keep]
        weights = w / w.sum()
        weighted_fedavg(global_model, active, weights)
        true_scores = ip_scores
        raw         = ip_raw

    elif method == "shapley":
        keep    = np.ones(n, dtype=bool)
        sv      = tmc_shapley(local_models, global_model, val_loader, device,
                              T=cfg.get("shapley_T", 20))
        weights = sv   # already normalised, non-negative
        weighted_fedavg(global_model, local_models, weights)

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
        f_byz   = max(1, int(round(cfg.get("byzantine_fraction", 0.2) * n)))
        keep    = krum_select(local_models, global_model, n_byzantine=f_byz)
        active  = [m for m, k in zip(local_models, keep) if k]
        weights = np.ones(len(active)) / len(active)
        weighted_fedavg(global_model, active, weights)

    return global_model, weights, true_scores[keep], raw, keep


# ---------------------------------------------------------------------------
# main comparison loop
# ---------------------------------------------------------------------------

def run_comparison(cfg, save_csv=True):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = cfg.get("dataset", "cifar10")

    build_model, build_client_loaders = _get_model_and_loader(dataset)

    print(f"\nDevice: {device} | Dataset: {dataset} | alpha_D={cfg['alpha_dirichlet']} | "
          f"Clients: {cfg['num_clients']} | Rounds: {cfg['num_rounds']}", flush=True)
    print("=" * 65, flush=True)

    # shared partition — identical for ALL methods
    client_loaders, val_loader, partitions, full_val_loader, targets = \
        build_client_loaders(
            alpha=cfg["alpha_dirichlet"],
            num_clients=cfg["num_clients"],
            batch_size=cfg["local_batch_size"])

    # independent contribution ground truth (volume + class diversity entropy)
    num_classes = len(np.unique(targets))
    data_gt     = compute_data_quality_gt(partitions, targets,
                                          num_classes=num_classes, gamma=0.5)
    print(f"  Data-quality GT (independent): {np.round(data_gt, 3)}", flush=True)

    methods  = cfg.get("methods", ["fedavg", "valscore", "cosinefilter",
                                    "fltrust", "krum", "ours"])
    trackers = {m: ConvergenceTracker() for m in methods}
    final_acc  = {}
    final_fair = {}

    for method in methods:
        print(f"\n{'='*65}", flush=True)
        print(f"  Method: {method.upper()}", flush=True)
        print(f"{'='*65}", flush=True)

        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        global_model = build_model().to(device)
        clients = [
            FLClient(i, global_model, client_loaders[i],
                     lr=cfg["lr"], local_epochs=cfg["local_epochs"],
                     device=device)
            for i in range(cfg["num_clients"])]

        n_clients          = cfg["num_clients"]
        cumulative_rewards = np.zeros(n_clients)
        cumulative_loo     = np.zeros(n_clients)
        grad_norm_history  = []
        loo_history        = []

        for rnd in range(1, cfg["num_rounds"] + 1):
            global_model, weights, poc_scores, raw, keep = run_method(
                method, clients, global_model, val_loader,
                full_val_loader, device, cfg, rng_seed=rnd)

            # full n-client reward vector: filtered clients receive 0
            full_reward = np.zeros(n_clients)
            full_reward[keep] = weights
            cumulative_rewards += full_reward * REWARD_PER_ROUND

            # fairness relative to INDEPENDENT data-quality ground truth
            fair = fairness_index(data_gt, full_reward + 1e-12)

            grad_norm_history.append(raw["grad_norm"].copy())

            if rnd % 10 == 0:
                local_models = [c.model for c in clients]
                loo = compute_loo_contributions(
                    local_models, global_model, val_loader, device)
                cumulative_loo += loo
                loo_history.append((rnd, loo.copy()))

            acc = accuracy(global_model, full_val_loader, device)
            trackers[method].log(rnd, acc, fair)

            if rnd % 10 == 0:
                print(f"  Round {rnd:3d} | Acc: {acc:.4f} | GT-Fairness: {fair:.4f}",
                      flush=True)

        trackers[method].grad_norm_history = grad_norm_history
        trackers[method].loo_history       = loo_history

        final_acc[method]  = trackers[method].accuracies[-1]
        final_fair[method] = np.mean(trackers[method].fairness)

        from scipy.stats import spearmanr
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, pval = spearmanr(cumulative_rewards, cumulative_loo)
        trackers[method].loo_spearman   = float(rho)  if not np.isnan(rho)  else 0.0
        trackers[method].loo_spearman_p = float(pval) if not np.isnan(pval) else 1.0
        trackers[method].cumulative_loo = cumulative_loo.copy()
        trackers[method].cumulative_rew = cumulative_rewards.copy()

    # summary
    print(f"\n{'='*75}", flush=True)
    print(f"  FINAL RESULTS  (alpha_D={cfg['alpha_dirichlet']}, {cfg['num_rounds']} rounds)", flush=True)
    print(f"  Fairness = Jain index relative to DATA-QUALITY ground truth", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"  {'Method':<14} {'Final Acc':>10} {'Avg GT-Fair':>13}", flush=True)
    print(f"  {'-'*40}", flush=True)
    for m in methods:
        marker = " <-- FairPoC" if m == "ours" else ""
        print(f"  {m.upper():<14} {final_acc[m]:>9.4f}  {final_fair[m]:>12.4f}{marker}",
              flush=True)
    print(f"{'='*75}\n", flush=True)

    if save_csv:
        os.makedirs("results", exist_ok=True)
        alpha_tag = str(cfg["alpha_dirichlet"]).replace(".", "")
        ds_tag    = f"_{dataset}" if dataset != "cifar10" else ""
        csv_path  = f"results/comparison{ds_tag}_alpha{alpha_tag}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round"] +
                            [f"{m}_acc"      for m in methods] +
                            [f"{m}_fairness" for m in methods])
            for i, rnd in enumerate(trackers["fedavg"].rounds):
                row  = [rnd]
                row += [trackers[m].accuracies[i] for m in methods]
                row += [trackers[m].fairness[i]   for m in methods]
                writer.writerow(row)
        print(f"  Saved: {csv_path}", flush=True)

        loo_path = csv_path.replace("comparison_", "loo_")
        with open(loo_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "loo_spearman_rho", "loo_spearman_p"])
            for m in methods:
                writer.writerow([m, trackers[m].loo_spearman,
                                    trackers[m].loo_spearman_p])

        stab_path = csv_path.replace("comparison_", "stability_")
        with open(stab_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "round"] +
                            [f"client_{i}" for i in range(n_clients)])
            for rnd_idx, gn in enumerate(trackers["ours"].grad_norm_history):
                writer.writerow(["grad_norm", rnd_idx + 1] + list(gn))
            for rnd, loo in trackers["ours"].loo_history:
                writer.writerow(["loo", rnd] + list(loo))

        # save GT vector for reference
        gt_path = csv_path.replace("comparison_", "gt_")
        np.savetxt(gt_path, data_gt, delimiter=",",
                   header="data_quality_gt", comments="")
        print(f"  GT saved: {gt_path}", flush=True)

    return trackers, final_acc, final_fair


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = {
        "num_clients": NUM_CLIENTS, "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS, "local_batch_size": LOCAL_BATCH_SIZE,
        "lr": LEARNING_RATE, "alpha_dirichlet": DIRICHLET_ALPHA,
        "alpha_poc": ALPHA_ACC, "beta_poc": BETA_DIV,
        "byzantine_fraction": 0.0, "byzantine_attack": "random", "seed": SEED,
        "dataset": "cifar10",
    }
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    run_comparison(cfg)
