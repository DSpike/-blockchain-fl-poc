"""
Multi-seed comparison for CIFAR-100 across all non-IID alpha values.
Runs 3 seeds × 4 alpha values = 12 configs sequentially.
Saves aggregated mean±std CSVs for paper Table III.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import numpy as np
from simulation.run_multiseed_comparison import run_multiseed
from config import (NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS,
                    LOCAL_BATCH_SIZE, LEARNING_RATE,
                    ALPHA_ACC, BETA_DIV)

SEEDS  = [42, 123, 456]
ALPHAS = [0.1, 0.5, 1.0, 10.0]

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
    "dataset":            "cifar100",
}

if __name__ == "__main__":
    for alpha in ALPHAS:
        cfg = copy.deepcopy(BASE_CFG)
        cfg["alpha_dirichlet"] = alpha
        print(f"\n{'='*65}")
        print(f"  CIFAR-100  alpha_D={alpha}  ({len(SEEDS)} seeds)")
        print(f"{'='*65}", flush=True)
        run_multiseed(cfg, seeds=SEEDS)
        print(f"  Done: alpha_D={alpha}", flush=True)
    print("\nAll CIFAR-100 configs complete.")
