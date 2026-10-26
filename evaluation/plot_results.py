"""
Generate all paper plots from GradNorm PoC runs.

Produces:
  results/plots/convergence_cifar10.png       — accuracy curves, 4 alphas
  results/plots/convergence_fashionmnist.png  — accuracy curves, 4 alphas
  results/plots/fairness_cifar10.png          — GT-fairness curves, 4 alphas
  results/plots/fairness_fashionmnist.png     — GT-fairness curves, 4 alphas
  results/plots/summary_bar.png               — final acc + fairness bar chart
  results/plots/combined_overview.png         — 2x4 grid (both datasets, acc+fair)
  results/plots/fairness_heatmap.png          — heatmap methods x alphas
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

METHODS = ["fedavg", "valscore", "cosinefilter", "ours"]
LABELS  = {"fedavg": "FedAVG", "valscore": "PoCL",
           "cosinefilter": "BLADE-FL", "ours": "FairPoC (Ours)"}
COLORS  = {"fedavg": "#999999", "valscore": "#4878CF",
           "cosinefilter": "#D65F5F", "ours": "#2ca02c"}
STYLES  = {"fedavg": "--", "valscore": "-.", "cosinefilter": ":", "ours": "-"}
WIDTHS  = {"fedavg": 1.4, "valscore": 1.4, "cosinefilter": 1.4, "ours": 2.2}

ALPHAS      = [0.1, 0.5, 1.0, 10.0]
ALPHA_TAGS  = ["01", "05", "10", "100"]
ALPHA_LABELS = {0.1: "alpha=0.1 (extreme non-IID)", 0.5: "alpha=0.5",
                1.0: "alpha=1.0", 10.0: "alpha=10.0 (near-IID)"}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_csv(dataset, alpha_tag):
    prefix = "" if dataset == "cifar10" else f"_{dataset}"
    path = os.path.join(RESULTS_DIR, f"comparison{prefix}_alpha{alpha_tag}.csv")
    return pd.read_csv(path)


def plot_curves(dataset, metric, ylabel, title_prefix, out_name, ylim=None):
    ds_label = "CIFAR-10" if dataset == "cifar10" else "FashionMNIST"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

    for ax, (alpha, tag) in zip(axes, zip(ALPHAS, ALPHA_TAGS)):
        df = load_csv(dataset, tag)
        for m in METHODS:
            col  = f"{m}_{metric}"
            vals = df[col].values * (100 if metric == "acc" else 1)
            ax.plot(df["round"], vals,
                    color=COLORS[m], linestyle=STYLES[m],
                    linewidth=WIDTHS[m], label=LABELS[m], alpha=0.9)

        ax.set_title(ALPHA_LABELS[alpha], fontsize=9)
        ax.set_xlabel("Round")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.suptitle(f"{title_prefix} — {ds_label}", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_summary_bar():
    datasets   = ["cifar10", "fashionmnist"]
    ds_labels  = ["CIFAR-10", "FashionMNIST"]
    fig, axes  = plt.subplots(2, 2, figsize=(14, 9))

    for row, (ds, ds_label) in enumerate(zip(datasets, ds_labels)):
        acc_data  = {m: [] for m in METHODS}
        fair_data = {m: [] for m in METHODS}
        for tag in ALPHA_TAGS:
            df = load_csv(ds, tag)
            for m in METHODS:
                acc_data[m].append(df[f"{m}_acc"].iloc[-1] * 100)
                fair_data[m].append(df[f"{m}_fairness"].iloc[-1])

        x     = np.arange(len(ALPHAS))
        width = 0.18

        for col, (data, ylabel, title) in enumerate([
            (acc_data,  "Final Accuracy (%)",     f"{ds_label} — Accuracy"),
            (fair_data, "GT-Fairness (Jain's J)", f"{ds_label} — Fairness"),
        ]):
            ax = axes[row][col]
            for i, m in enumerate(METHODS):
                bars = ax.bar(x + (i - 1.5) * width, data[m], width,
                              label=LABELS[m], color=COLORS[m], alpha=0.85,
                              edgecolor="white", linewidth=0.5)
                if m == "ours":
                    for bar in bars:
                        bar.set_edgecolor("#1a7a1a")
                        bar.set_linewidth(1.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"a={a}" for a in ALPHAS])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax.legend(fontsize=8)

    plt.suptitle("FairPoC vs Baselines — Final Results Summary\n(GradNorm PoC, 50 rounds, seed=42)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "summary_bar.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_combined_overview():
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.3)

    datasets  = ["cifar10", "fashionmnist"]
    ds_labels = ["CIFAR-10", "FashionMNIST"]

    for row, (ds, ds_label) in enumerate(zip(datasets, ds_labels)):
        for col, (alpha, tag) in enumerate(zip(ALPHAS, ALPHA_TAGS)):
            ax  = fig.add_subplot(gs[row, col])
            df  = load_csv(ds, tag)
            ax2 = ax.twinx()

            for m in METHODS:
                ax.plot(df["round"], df[f"{m}_acc"].values * 100,
                        color=COLORS[m], linestyle=STYLES[m],
                        linewidth=WIDTHS[m], alpha=0.9)
                ax2.plot(df["round"], df[f"{m}_fairness"].values,
                         color=COLORS[m], linestyle=STYLES[m],
                         linewidth=WIDTHS[m], alpha=0.35)

            ax.set_title(f"{ds_label} | {ALPHA_LABELS[alpha]}", fontsize=8)
            ax.set_xlabel("Round", fontsize=8)
            if col == 0:
                ax.set_ylabel("Accuracy (%)", fontsize=8)
            if col == 3:
                ax2.set_ylabel("GT-Fairness", fontsize=8, color="gray")
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.25)

    handles = [plt.Line2D([0], [0], color=COLORS[m], linestyle=STYLES[m],
                          linewidth=WIDTHS[m], label=LABELS[m])
               for m in METHODS]
    fig.legend(handles, [LABELS[m] for m in METHODS],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04),
               frameon=True, fontsize=9)
    fig.suptitle("FairPoC — Convergence Overview  (solid line=accuracy, faded=GT-fairness)",
                 fontsize=12, y=1.01)
    out = os.path.join(PLOTS_DIR, "combined_overview.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fairness_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    datasets  = ["cifar10", "fashionmnist"]
    ds_labels = ["CIFAR-10", "FashionMNIST"]

    for ax, ds, ds_label in zip(axes, datasets, ds_labels):
        matrix = np.zeros((len(METHODS), len(ALPHAS)))
        for j, tag in enumerate(ALPHA_TAGS):
            df = load_csv(ds, tag)
            for i, m in enumerate(METHODS):
                matrix[i, j] = df[f"{m}_fairness"].iloc[-1]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels([f"a={a}" for a in ALPHAS])
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([LABELS[m] for m in METHODS])
        ax.set_title(f"{ds_label} — GT-Fairness (final round)")

        for i in range(len(METHODS)):
            for j in range(len(ALPHAS)):
                val = matrix[i, j]
                txt_color = "white" if val < 0.75 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=txt_color,
                        fontweight="bold" if METHODS[i] == "ours" else "normal")

        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("GT-Fairness Heatmap — FairPoC vs Baselines\n(green=high, red=low; FairPoC=bottom row)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "fairness_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("\nGenerating all plots...\n")

    plot_curves("cifar10", "acc", "Accuracy (%)", "Convergence",
                "convergence_cifar10.png", ylim=(10, 95))
    plot_curves("fashionmnist", "acc", "Accuracy (%)", "Convergence",
                "convergence_fashionmnist.png", ylim=(60, 97))

    plot_curves("cifar10", "fairness", "GT-Fairness (Jain's J)",
                "GT-Fairness", "fairness_cifar10.png", ylim=(0.4, 1.02))
    plot_curves("fashionmnist", "fairness", "GT-Fairness (Jain's J)",
                "GT-Fairness", "fairness_fashionmnist.png", ylim=(0.4, 1.02))

    plot_summary_bar()
    plot_combined_overview()
    plot_fairness_heatmap()

    print("\nDone. All plots saved to results/plots/")
