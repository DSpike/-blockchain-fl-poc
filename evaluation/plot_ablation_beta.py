"""
Plot beta_poc ablation results from results/ablation_beta.csv.

Generates:
  - results/ablation_beta.pdf  (accuracy vs fairness trade-off for beta sweep)
  - results/ablation_beta_acc.pdf  (accuracy vs beta)
  - results/ablation_beta_fair.pdf (GT-Fairness vs beta)

Usage:
    cd blockchain-fl-poc
    python evaluation/plot_ablation_beta.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

IN_PATH   = "results/ablation_beta.csv"
OUT_DIR   = "results"
ALPHA_VALS = [0.1, 0.5, 1.0, 10.0]
COLORS     = {0.1: "#d62728", 0.5: "#ff7f0e",
              1.0: "#1f77b4", 10.0: "#2ca02c"}
LABELS     = {0.1:  r"$\alpha_D=0.1$ (hard non-IID)",
              0.5:  r"$\alpha_D=0.5$",
              1.0:  r"$\alpha_D=1.0$ (moderate non-IID)",
              10.0: r"$\alpha_D=10.0$ (near-IID)"}
OPT_BETA   = 0.3   # the β we use in the paper — draw a vertical reference


def main():
    if not os.path.exists(IN_PATH):
        print(f"ERROR: {IN_PATH} not found. Run run_ablation_beta.py first.")
        sys.exit(1)

    df = pd.read_csv(IN_PATH)
    print(df.to_string())

    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       10,
        "axes.titlesize":  11,
        "axes.labelsize":  11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # ---- figure 2: side-by-side acc and fairness vs beta (main paper figure) ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for alpha in ALPHA_VALS:
        sub   = df[df["alpha_D"] == alpha].sort_values("beta_poc")
        betas = sub["beta_poc"].values
        accs  = sub["final_acc"].values * 100
        fair  = sub["final_gt_fairness"].values

        axes[0].plot(betas, accs, "o-", color=COLORS[alpha], label=LABELS[alpha],
                     linewidth=1.5, markersize=5)
        axes[1].plot(betas, fair, "o-", color=COLORS[alpha], label=LABELS[alpha],
                     linewidth=1.5, markersize=5)

    for ax, ylabel, title in zip(
        axes,
        ["Final Accuracy (%)", "Avg. GT-Fairness (Jain Index)"],
        [r"(a) Accuracy vs. $\beta_p$", r"(b) GT-Fairness vs. $\beta_p$"],
    ):
        ax.axvline(OPT_BETA, color="black", linestyle="--", linewidth=1.0,
                   label=r"Selected $\beta_p=0.3$")
        ax.set_xlabel(r"Gradient Norm Weight $\beta_p$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    fig.tight_layout()
    out2 = os.path.join(OUT_DIR, "ablation_beta_sensitivity.pdf")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    fig.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig)

    # ---- print table for paper ----
    print("\n--- Table for paper (CIFAR-10, FairPoC only) ---")
    print(f"{'beta':>5} | {'alpha_D':>8} | {'Acc%':>7} | {'GT-Fair':>8}")
    print("-" * 40)
    for _, row in df.sort_values(["alpha_D", "beta_poc"]).iterrows():
        print(f"  {row['beta_poc']:.1f}  | {row['alpha_D']:>8}  | "
              f"{row['final_acc']*100:>6.2f}  | {row['final_gt_fairness']:>7.4f}")


if __name__ == "__main__":
    main()
