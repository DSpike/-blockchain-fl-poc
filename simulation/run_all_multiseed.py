"""
Run multi-seed validation for all four alpha values sequentially.
Produces: results/multiseed_alpha{tag}_5seeds.csv for each alpha.

Usage:
  python simulation/run_all_multiseed.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.run_multiseed import run_multiseed
from config import *

ALPHAS   = [0.1, 0.5, 1.0, 10.0]
N_SEEDS  = 3

BASE_CFG = {
    "num_clients":      NUM_CLIENTS,
    "num_rounds":       50,
    "local_epochs":     LOCAL_EPOCHS,
    "local_batch_size": LOCAL_BATCH_SIZE,
    "lr":               LEARNING_RATE,
    "alpha_poc":        ALPHA_ACC,
    "beta_poc":         BETA_DIV,
    "byzantine_fraction": 0.0,
    "byzantine_attack": "random",
    "seed":             0,
}

if __name__ == "__main__":
    for alpha in ALPHAS:
        cfg = dict(BASE_CFG)
        cfg["alpha_dirichlet"] = alpha
        print(f"\n{'#'*65}")
        print(f"  STARTING MULTISEED: alpha={alpha}, seeds={N_SEEDS}")
        print(f"{'#'*65}\n")
        run_multiseed(cfg, n_seeds=N_SEEDS)

    print("\nAll multiseed runs complete.")
    print("Results saved in results/multiseed_alpha*_5seeds.csv")
