"""
Multi-seed comparison runner.
Runs run_comparison() with seeds 42 / 123 / 456 for every
(dataset, alpha) combination and saves per-seed CSVs plus
an aggregated mean±std CSV used by the paper tables.

Usage:
    cd blockchain-fl-poc
    python simulation/run_multiseed_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv, copy
import numpy as np
from simulation.run_comparison import run_comparison
from config import (NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS,
                    LOCAL_BATCH_SIZE, LEARNING_RATE, ALPHA_ACC, BETA_DIV)

SEEDS    = [42, 123, 456]
ALPHAS   = [0.1, 0.5, 1.0, 10.0]
DATASETS = ["cifar10", "fashionmnist", "cifar100", "svhn"]
METHODS  = ["fedavg", "valscore", "cosinefilter", "fltrust", "krum", "ours"]

os.makedirs("results/seeds", exist_ok=True)

BASE_CFG = {
    "num_clients":      NUM_CLIENTS,
    "num_rounds":       NUM_ROUNDS,
    "local_epochs":     LOCAL_EPOCHS,
    "local_batch_size": LOCAL_BATCH_SIZE,
    "lr":               LEARNING_RATE,
    "alpha_poc":        ALPHA_ACC,
    "beta_poc":         BETA_DIV,
    "byzantine_fraction": 0.0,
    "byzantine_attack": "random",
}


def alpha_tag(a):
    return str(a).replace(".", "")


def ds_tag(ds):
    return f"_{ds}" if ds != "cifar10" else ""


def run_all():
    for dataset in DATASETS:
        for alpha in ALPHAS:
            tag  = alpha_tag(alpha)
            dtag = ds_tag(dataset)
            print(f"\n{'#'*70}")
            print(f"#  {dataset.upper()}  alpha={alpha}")
            print(f"{'#'*70}")

            # collect per-round data across seeds
            # shape: (n_seeds, n_rounds, n_methods)
            acc_seeds  = []
            fair_seeds = []
            rounds_ref = None

            for seed in SEEDS:
                cfg = copy.deepcopy(BASE_CFG)
                cfg["alpha_dirichlet"] = alpha
                cfg["seed"]            = seed
                cfg["dataset"]         = dataset

                trackers, final_acc, final_fair = run_comparison(
                    cfg, save_csv=False)

                rounds = trackers["fedavg"].rounds
                if rounds_ref is None:
                    rounds_ref = rounds

                acc_row  = [[trackers[m].accuracies[i] for m in METHODS]
                            for i in range(len(rounds))]
                fair_row = [[trackers[m].fairness[i]   for m in METHODS]
                            for i in range(len(rounds))]
                acc_seeds.append(acc_row)
                fair_seeds.append(fair_row)

                # save per-seed CSV
                seed_path = f"results/seeds/comparison{dtag}_alpha{tag}_seed{seed}.csv"
                with open(seed_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["round"] +
                                    [f"{m}_acc"      for m in METHODS] +
                                    [f"{m}_fairness" for m in METHODS])
                    for i, rnd in enumerate(rounds):
                        row  = [rnd]
                        row += [trackers[m].accuracies[i] for m in METHODS]
                        row += [trackers[m].fairness[i]   for m in METHODS]
                        writer.writerow(row)
                print(f"  Seed {seed} saved: {seed_path}", flush=True)

            # aggregate: mean and std across seeds
            acc_arr  = np.array(acc_seeds)   # (n_seeds, n_rounds, n_methods)
            fair_arr = np.array(fair_seeds)

            acc_mean  = acc_arr.mean(axis=0)
            acc_std   = acc_arr.std(axis=0)
            fair_mean = fair_arr.mean(axis=0)
            fair_std  = fair_arr.std(axis=0)

            agg_path = f"results/comparison_ms{dtag}_alpha{tag}.csv"
            with open(agg_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["round"]
                for m in METHODS:
                    header += [f"{m}_acc_mean", f"{m}_acc_std",
                               f"{m}_fair_mean", f"{m}_fair_std"]
                writer.writerow(header)
                for i, rnd in enumerate(rounds_ref):
                    row = [rnd]
                    for j, m in enumerate(METHODS):
                        row += [acc_mean[i,j], acc_std[i,j],
                                fair_mean[i,j], fair_std[i,j]]
                    writer.writerow(row)
            print(f"  Aggregated (mean±std over {len(SEEDS)} seeds): {agg_path}",
                  flush=True)


if __name__ == "__main__":
    run_all()
    print("\nAll multi-seed runs complete.")
