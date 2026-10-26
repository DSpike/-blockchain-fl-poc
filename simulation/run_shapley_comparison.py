"""
Shapley baseline comparison.

Runs FedAVG, FairPoC, and TMC-Shapley on two representative configurations:
  - CIFAR-10 alpha_D = 0.1  (hard non-IID, the challenging case)
  - CIFAR-10 alpha_D = 1.0  (moderate non-IID)

Reports final accuracy, GT-Fairness, and per-round wall-clock time
(to show computational overhead of Shapley vs. FairPoC).

Usage:
    cd blockchain-fl-poc
    python simulation/run_shapley_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv, copy, time
import numpy as np
from simulation.run_comparison import run_comparison
from config import (NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS,
                    LOCAL_BATCH_SIZE, LEARNING_RATE, ALPHA_ACC, BETA_DIV)

SEED     = 42
ALPHAS   = [0.1, 1.0]
METHODS  = ["fedavg", "ours", "shapley"]   # focused comparison

BASE_CFG = {
    "num_clients":        NUM_CLIENTS,
    "num_rounds":         NUM_ROUNDS,
    "local_epochs":       LOCAL_EPOCHS,
    "local_batch_size":   LOCAL_BATCH_SIZE,
    "lr":                 LEARNING_RATE,
    "alpha_poc":          ALPHA_ACC,
    "beta_poc":           BETA_DIV,
    "byzantine_fraction": 0.0,
    "byzantine_attack":   "random",
    "seed":               SEED,
    "dataset":            "cifar10",
    "methods":            METHODS,
    "shapley_T":          20,     # MC samples per round
}

os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    rows = []  # (alpha, method, final_acc, final_fair, round_time)

    for alpha in ALPHAS:
        cfg = copy.deepcopy(BASE_CFG)
        cfg["alpha_dirichlet"] = alpha

        print(f"\n{'#'*65}")
        print(f"#  Shapley comparison — CIFAR-10  alpha_D={alpha}")
        print(f"{'#'*65}")

        t0 = time.time()
        trackers, final_acc, final_fair = run_comparison(cfg, save_csv=False)
        elapsed = time.time() - t0
        round_time = elapsed / NUM_ROUNDS  # average seconds per round

        print(f"\n  Wall-clock: {elapsed:.1f}s  ({round_time:.2f}s/round avg)")
        for m in METHODS:
            print(f"  {m:10s}  acc={final_acc[m]*100:.2f}%  "
                  f"GT-fair={final_fair[m]:.4f}")
            rows.append((alpha, m, final_acc[m], final_fair[m], round_time))

    # save summary
    out_path = "results/shapley_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha_D", "method", "final_acc", "final_gt_fairness",
                         "avg_round_time_s"])
        for row in rows:
            writer.writerow(row)

    print(f"\nResults saved: {out_path}")
    print("\nalpha | method    | acc%  | GT-Fair | s/round")
    print("-" * 52)
    for alpha, m, acc, fair, rt in rows:
        print(f"  {alpha:4} | {m:9s} | {acc*100:.2f}% | {fair:.4f}  | {rt:.2f}s")
