"""
Run ablation, multiseed, and shapley experiments sequentially so each
gets exclusive GPU access. Launch with:

    python simulation/run_all_sequential.py
"""

import subprocess, sys, time

JOBS = [
    ("Ablation-β",   ["python", "-u", "simulation/run_ablation_beta.py"],
                     "results/ablation_beta.log",  "results/ablation_beta_err.log"),
    ("Multi-seed",   ["python", "-u", "simulation/run_multiseed_comparison.py"],
                     "results/multiseed.log",       "results/multiseed_err.log"),
    ("Shapley",      ["python", "-u", "simulation/run_shapley_comparison.py"],
                     "results/shapley.log",          "results/shapley_err.log"),
]

for name, cmd, out_path, err_path in JOBS:
    print(f"\n{'='*60}", flush=True)
    print(f"  Starting: {name}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    with open(out_path, "w") as fout, open(err_path, "w") as ferr:
        proc = subprocess.run(cmd, stdout=fout, stderr=ferr)
    elapsed = (time.time() - t0) / 60
    status = "OK" if proc.returncode == 0 else f"EXIT {proc.returncode}"
    print(f"  {name} done in {elapsed:.1f} min  [{status}]", flush=True)

print("\nAll jobs complete.")
