"""
Run comparison experiment on FashionMNIST across all 4 alpha values.
Produces: results/comparison_fashionmnist_alpha*.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.run_comparison import run_comparison
from config import *

ALPHAS = [0.1, 0.5, 1.0, 10.0]

BASE_CFG = {
    "dataset":          "fashionmnist",
    "num_clients":      NUM_CLIENTS,
    "num_rounds":       50,
    "local_epochs":     LOCAL_EPOCHS,
    "local_batch_size": LOCAL_BATCH_SIZE,
    "lr":               LEARNING_RATE,
    "alpha_poc":        ALPHA_ACC,
    "beta_poc":         BETA_DIV,
    "byzantine_fraction": 0.0,
    "byzantine_attack": "random",
    "seed":             42,
}

if __name__ == "__main__":
    for alpha in ALPHAS:
        cfg = dict(BASE_CFG)
        cfg["alpha_dirichlet"] = alpha
        print(f"\n{'#'*65}")
        print(f"  FashionMNIST | alpha={alpha}")
        print(f"{'#'*65}\n")
        run_comparison(cfg, save_csv=True)

    print("\nAll FashionMNIST runs complete.")
    print("Results in results/comparison_fashionmnist_alpha*.csv")
