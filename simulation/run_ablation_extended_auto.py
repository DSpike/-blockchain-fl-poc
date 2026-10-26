"""
Waits for the shapley run to complete, then runs the ablation-beta
sweep for MISSING alpha values only (0.5 and 10.0 — alpha=0.1 and
alpha=1.0 already exist in ablation_beta.csv). Then generates the
final 4-curve figure automatically.
"""

import sys, os, time, subprocess

TRIGGER  = "results/shapley_comparison.csv"
POLL_SEC = 60

print("Waiting for shapley run to complete...", flush=True)
while not os.path.exists(TRIGGER):
    time.sleep(POLL_SEC)

print(f"\nShapley done — running missing ablation configs (alpha=0.5, 10.0).", flush=True)
print(f"14 configs total (~17 hours).\n", flush=True)

result = subprocess.run(
    ["python", "-u", "simulation/run_ablation_beta_missing.py"],
    stdout=sys.stdout, stderr=sys.stderr
)

if result.returncode == 0:
    print("\nAblation done. Generating final 4-curve plot...", flush=True)
    subprocess.run(["python", "-u", "evaluation/plot_ablation_beta.py"],
                   stdout=sys.stdout, stderr=sys.stderr)
    print("Plot saved to results/ablation_beta_sensitivity.png", flush=True)
else:
    print(f"\nAblation failed with exit code {result.returncode}", flush=True)
