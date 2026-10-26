"""Preview ablation-beta plot using alpha=0.1 results only (alpha=1.0 still running)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

df = pd.read_csv("results/ablation_beta_preview.csv")
sub   = df[df["alpha_D"] == 0.1].sort_values("beta_poc")
betas = sub["beta_poc"].values
accs  = sub["final_acc"].values * 100
fair  = sub["final_gt_fairness"].values

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

for ax, y, ylabel, title in zip(
    axes,
    [accs, fair],
    ["Final Accuracy (%)", "Avg. GT-Fairness (Jain Index)"],
    ["(a) Accuracy vs. $\\beta_p$", "(b) GT-Fairness vs. $\\beta_p$"],
):
    ax.plot(betas, y, "o-", color="#d62728", linewidth=1.5, markersize=5,
            label=r"FairPoC ($\alpha_D=0.1$, hard non-IID)")
    ax.axvline(0.3, color="black", linestyle="--", linewidth=1.0,
               label=r"Selected $\beta_p = 0.3$")
    ax.set_xlabel(r"Gradient Norm Weight $\beta_p$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

fig.tight_layout()
out = "results/ablation_beta_preview.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
