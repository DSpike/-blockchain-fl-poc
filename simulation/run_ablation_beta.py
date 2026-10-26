"""
Ablation: sweep beta_poc (diversity weight) at fixed alpha_D=0.1 and 1.0
on CIFAR-10. Reports final accuracy and GT-Fairness for FairPoC only.

Usage:
    cd blockchain-fl-poc
    python simulation/run_ablation_beta.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv, copy
import numpy as np
from simulation.run_comparison import run_comparison
from config import (NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS,
                    LOCAL_BATCH_SIZE, LEARNING_RATE)

BETAS    = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
ALPHAS   = [0.1, 0.5, 1.0, 10.0]
SEED     = 42

os.makedirs("results", exist_ok=True)

BASE_CFG = {
    "num_clients":      NUM_CLIENTS,
    "num_rounds":       NUM_ROUNDS,
    "local_epochs":     LOCAL_EPOCHS,
    "local_batch_size": LOCAL_BATCH_SIZE,
    "lr":               LEARNING_RATE,
    "byzantine_fraction": 0.0,
    "byzantine_attack": "random",
    "seed":             SEED,
    "dataset":          "cifar10",
}

if __name__ == "__main__":
    results = []  # list of (alpha, beta, final_acc, final_fair)

    for alpha in ALPHAS:
        for beta in BETAS:
            alpha_val = 1.0 - beta
            cfg = copy.deepcopy(BASE_CFG)
            cfg["alpha_dirichlet"] = alpha
            cfg["alpha_poc"]       = alpha_val
            cfg["beta_poc"]        = beta

            print(f"\nalpha_D={alpha}  beta_poc={beta:.1f}  (acc_weight={alpha_val:.1f})",
                  flush=True)

            trackers, final_acc, final_fair = run_comparison(cfg, save_csv=False)

            acc  = final_acc["ours"]
            fair = final_fair["ours"]
            results.append((alpha, beta, alpha_val, acc, fair))
            print(f"  FairPoC: acc={acc:.4f}  GT-Fair={fair:.4f}", flush=True)

    # save
    out_path = "results/ablation_beta.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha_D", "beta_poc", "alpha_poc",
                         "final_acc", "final_gt_fairness"])
        for row in results:
            writer.writerow(row)

    print(f"\nAblation results saved: {out_path}")
    print("\nalpha_D | beta | acc_weight | final_acc | GT-Fairness")
    print("-" * 60)
    for alpha, beta, alpha_val, acc, fair in results:
        print(f"  {alpha:4}  |  {beta:.1f} |   {alpha_val:.1f}      |  {acc*100:.2f}%  |  {fair:.4f}")
