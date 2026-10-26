"""
Run comparison for all 4 alpha values with save_csv=True.
Generates stability_alpha*.csv files needed for the stability plot.
One seed (42) is sufficient — grad_div stability is a structural property.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.run_comparison import run_comparison
from config import *

ALPHAS = [0.1, 0.5, 1.0, 10.0]

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
    "seed":             42,
}

if __name__ == "__main__":
    for alpha in ALPHAS:
        cfg = dict(BASE_CFG)
        cfg["alpha_dirichlet"] = alpha
        print(f"\n{'#'*65}")
        print(f"  STABILITY RUN: alpha={alpha}")
        print(f"{'#'*65}\n")
        run_comparison(cfg, save_csv=True)

    print("\nAll stability runs complete.")
    print("Now run: python evaluation/plot_stability.py")
