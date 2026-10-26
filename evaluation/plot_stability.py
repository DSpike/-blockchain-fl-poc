"""
Gradient divergence stability vs LOO signal degradation.

Key result: gradient divergence scores remain stable and differentiated
across training rounds, while LOO contributions collapse near convergence
(PoShapley-BCFL's acknowledged limitation).

Generates: results/plots/stability_grad_div_vs_loo.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

CLIENT_COLORS = [
    "#4878CF", "#D65F5F", "#6ACC65", "#B47CC7",
    "#C4AD66", "#77BEDB", "#E07000", "#60BD68",
    "#F15854", "#A09F00",
]


def plot_stability(csv_path, out_dir, alpha_tag):
    df = pd.read_csv(csv_path)

    grad_df = df[df["type"] == "grad_norm"].copy()
    loo_df  = df[df["type"] == "loo"].copy()

    client_cols = [c for c in df.columns if c.startswith("client_")]
    n_clients   = len(client_cols)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Left: gradient divergence per client across rounds ──────────────────
    ax = axes[0]
    rounds_gd = grad_df["round"].values
    for i, col in enumerate(client_cols):
        ax.plot(rounds_gd, grad_df[col].values,
                color=CLIENT_COLORS[i % len(CLIENT_COLORS)],
                linewidth=1.3, alpha=0.85,
                label=f"Client {i}")

    ax.set_xlabel("Training Round")
    ax.set_ylabel("Normalised Gradient Magnitude")
    ax.set_title("FairPoC: Gradient Magnitude\n(stable, positively correlated with data quality)")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    # annotation: stable
    ax.annotate("Signal remains\nstable & ordered by data quality",
                xy=(rounds_gd[-1] * 0.6, grad_df[client_cols].values.mean() * 1.3),
                fontsize=9, color="#007700",
                bbox=dict(boxstyle="round,pad=0.3", fc="#CCFFCC", ec="#007700", alpha=0.8))

    # ── Right: LOO contribution per client across rounds ────────────────────
    ax = axes[1]
    if len(loo_df) > 0:
        rounds_loo = loo_df["round"].values
        for i, col in enumerate(client_cols):
            ax.plot(rounds_loo, loo_df[col].values,
                    color=CLIENT_COLORS[i % len(CLIENT_COLORS)],
                    linewidth=1.3, alpha=0.85, marker="o", markersize=3,
                    label=f"Client {i}")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xlabel("Training Round")
        ax.set_ylabel("LOO Contribution (Δ Accuracy)")
        ax.set_title("LOO Contribution Signal\n(degrades as model converges)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=7, ncol=2)

        # annotation: degrading
        ax.annotate("Signal collapses\nnear convergence",
                    xy=(rounds_loo[len(rounds_loo)//2],
                        loo_df[client_cols].values.max() * 0.6),
                    fontsize=9, color="#CC0000",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFCCCC", ec="#CC0000", alpha=0.8))
    else:
        axes[1].text(0.5, 0.5, "LOO data not available",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle(
        f"Incentive Signal Stability: Gradient Magnitude vs LOO  (α={alpha_tag})\n"
        "FairPoC's gradient magnitude is stable at convergence; LOO collapses (PoShapley-BCFL limitation)",
        fontsize=11, y=1.03
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"stability_grad_norm_vs_loo_alpha{alpha_tag}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main(results_dir="results", out_dir="results/plots"):
    import glob
    TAG = {"01": "0.1", "05": "0.5", "10": "1.0", "100": "10.0"}
    files = sorted(glob.glob(os.path.join(results_dir, "stability_alpha*.csv")))
    if not files:
        print("No stability CSVs found. Run run_comparison.py first.")
        return
    for f in files:
        tag = os.path.basename(f).replace("stability_alpha", "").replace(".csv", "")
        alpha_label = TAG.get(tag, tag)
        print(f"  Plotting alpha={alpha_label} ...")
        plot_stability(f, out_dir, alpha_label)
    print(f"\nDone. Plots in {out_dir}/")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    main(
        results_dir=os.path.join(base, "results"),
        out_dir=os.path.join(base, "results", "plots"),
    )
