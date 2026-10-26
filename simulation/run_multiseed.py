"""
Multi-seed runner: runs comparison across N seeds and computes mean ± std.
Usage:
  python simulation/run_multiseed.py --config experiments/exp1_iid_vs_noniid.yaml --seeds 5
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse, yaml, csv
import numpy as np
from simulation.run_comparison import run_comparison

METHODS = ["fedavg", "pocl", "blade_fl", "ours"]


def run_multiseed(cfg, n_seeds=5):
    seeds = list(range(n_seeds))
    alpha_tag = str(cfg["alpha_dirichlet"]).replace(".", "")

    all_acc   = {m: [] for m in METHODS}
    all_fair  = {m: [] for m in METHODS}

    for seed in seeds:
        print(f"\n{'#'*65}", flush=True)
        print(f"  SEED {seed}  (α={cfg['alpha_dirichlet']})", flush=True)
        print(f"{'#'*65}", flush=True)

        cfg["seed"] = seed
        _, final_acc, final_fair = run_comparison(cfg, save_csv=False)

        for m in METHODS:
            all_acc[m].append(final_acc[m])
            all_fair[m].append(final_fair[m])

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}", flush=True)
    print(f"  MULTI-SEED RESULTS  (α={cfg['alpha_dirichlet']}, {n_seeds} seeds)", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'Method':<12} {'Acc mean':>10} {'Acc std':>9} {'Fair mean':>11} {'Fair std':>9}", flush=True)
    print(f"  {'-'*55}", flush=True)

    rows = []
    for m in METHODS:
        acc_mean  = np.mean(all_acc[m])  * 100
        acc_std   = np.std(all_acc[m])   * 100
        fair_mean = np.mean(all_fair[m])
        fair_std  = np.std(all_fair[m])
        marker = " ◄" if m == "ours" else ""
        print(f"  {m.upper():<12} {acc_mean:>9.2f}%  {acc_std:>7.2f}%  "
              f"{fair_mean:>10.4f}  {fair_std:>8.4f}{marker}", flush=True)
        rows.append([m, acc_mean, acc_std, fair_mean, fair_std])

    print(f"{'='*65}\n", flush=True)

    # ── save CSV ──────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = f"results/multiseed_alpha{alpha_tag}_{n_seeds}seeds.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "acc_mean_%", "acc_std_%", "fair_mean", "fair_std"])
        writer.writerows(rows)
    print(f"  Saved: {out_path}", flush=True)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seeds",  type=int, default=5)
    args = parser.parse_args()

    from config import *
    cfg = {
        "num_clients": NUM_CLIENTS, "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS, "local_batch_size": LOCAL_BATCH_SIZE,
        "lr": LEARNING_RATE, "alpha_dirichlet": DIRICHLET_ALPHA,
        "alpha_poc": ALPHA_ACC, "beta_poc": BETA_DIV,
        "byzantine_fraction": 0.0, "byzantine_attack": "random", "seed": 0,
    }
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    run_multiseed(cfg, n_seeds=args.seeds)
