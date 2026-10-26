"""
Supplementary figures — regenerated with GradNorm PoC results.
Replaces all June-30 stale plots with July-10 data.

Generates:
  convergence_alpha{tag}.png       — per-alpha accuracy curves (CIFAR-10)
  fairness_alpha{tag}.png          — per-alpha GT-fairness curves (CIFAR-10)
  fairness_bar_alpha{tag}.png      — per-alpha GT-fairness bar (CIFAR-10)
  comparison_table.png             — heatmap: both datasets, both metrics
  final_accuracy_bar.png           — final accuracy bar (CIFAR-10)
  final_fairness_bar.png           — final GT-fairness bar (CIFAR-10)
  fairness_gap_bar.png             — fairness gap vs FedAVG baseline
  data_distribution_alpha01.png    — data partition heatmap (unchanged)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

METHODS  = ["fedavg", "valscore", "cosinefilter", "ours"]
LABELS   = {"fedavg": "FedAVG", "valscore": "PoCL",
            "cosinefilter": "BLADE-FL", "ours": "FairPoC (Ours)"}
COLORS   = {"fedavg": "#999999", "valscore": "#4878CF",
            "cosinefilter": "#D65F5F", "ours": "#2ca02c"}
STYLES   = {"fedavg": "--", "valscore": "-.", "cosinefilter": ":", "ours": "-"}
WIDTHS   = {"fedavg": 1.4, "valscore": 1.4, "cosinefilter": 1.4, "ours": 2.2}

ALPHAS      = [0.1, 0.5, 1.0, 10.0]
ALPHA_TAGS  = ["01", "05", "10", "100"]
ALPHA_STR   = {0.1: "0.1", 0.5: "0.5", 1.0: "1.0", 10.0: "10.0"}

BASE    = os.path.dirname(os.path.dirname(__file__))
RES_DIR = os.path.join(BASE, "results")
OUT_DIR = os.path.join(RES_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def load(dataset, tag):
    prefix = "" if dataset == "cifar10" else f"_{dataset}"
    return pd.read_csv(os.path.join(RES_DIR, f"comparison{prefix}_alpha{tag}.csv"))


# ── Per-alpha convergence (accuracy) ────────────────────────────────────────

def plot_convergence_per_alpha(dataset="cifar10"):
    ds_label = "CIFAR-10" if dataset == "cifar10" else "FashionMNIST"
    for alpha, tag in zip(ALPHAS, ALPHA_TAGS):
        df  = load(dataset, tag)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in METHODS:
            ax.plot(df["round"], df[f"{m}_acc"] * 100,
                    color=COLORS[m], linestyle=STYLES[m],
                    linewidth=WIDTHS[m], label=LABELS[m], alpha=0.9)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"Convergence — {ds_label}  (Dirichlet α={ALPHA_STR[alpha]})")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.3)
        suffix = f"_{dataset}" if dataset != "cifar10" else ""
        path = os.path.join(OUT_DIR, f"convergence{suffix}_alpha{tag}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Per-alpha GT-fairness curves ─────────────────────────────────────────────

def plot_fairness_per_alpha(dataset="cifar10"):
    ds_label = "CIFAR-10" if dataset == "cifar10" else "FashionMNIST"
    for alpha, tag in zip(ALPHAS, ALPHA_TAGS):
        df  = load(dataset, tag)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in METHODS:
            ax.plot(df["round"], df[f"{m}_fairness"],
                    color=COLORS[m], linestyle=STYLES[m],
                    linewidth=WIDTHS[m], label=LABELS[m], alpha=0.9)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.4)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("GT-Fairness (Jain's J)")
        ax.set_title(f"GT-Fairness — {ds_label}  (Dirichlet α={ALPHA_STR[alpha]})")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim(0.4, 1.05)
        suffix = f"_{dataset}" if dataset != "cifar10" else ""
        path = os.path.join(OUT_DIR, f"fairness{suffix}_alpha{tag}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Per-alpha GT-fairness bar chart ─────────────────────────────────────────

def plot_fairness_bar_per_alpha(dataset="cifar10"):
    ds_label = "CIFAR-10" if dataset == "cifar10" else "FashionMNIST"
    for alpha, tag in zip(ALPHAS, ALPHA_TAGS):
        df  = load(dataset, tag)
        avgs = [df[f"{m}_fairness"].mean() for m in METHODS]
        mins = [df[f"{m}_fairness"].min()  for m in METHODS]
        x    = np.arange(len(METHODS))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(x, avgs, 0.45,
                      color=[COLORS[m] for m in METHODS],
                      alpha=0.85, edgecolor="white")
        ax.scatter(x, mins, color="black", zorder=5, s=40,
                   marker="v", label="Min GT-Fairness")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.1, alpha=0.45)

        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=10)
        ax.set_ylabel("GT-Fairness (Jain's J)")
        ax.set_title(f"Avg GT-Fairness — {ds_label}  α={ALPHA_STR[alpha]}  (50 rounds)")
        ax.set_ylim(max(0.4, min(avgs) - 0.05), 1.012)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="lower right")

        suffix = f"_{dataset}" if dataset != "cifar10" else ""
        path = os.path.join(OUT_DIR, f"fairness_bar{suffix}_alpha{tag}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Final accuracy & fairness bar charts ────────────────────────────────────

def plot_final_bars(dataset="cifar10"):
    ds_label = "CIFAR-10" if dataset == "cifar10" else "FashionMNIST"
    acc_final  = {m: [] for m in METHODS}
    fair_final = {m: [] for m in METHODS}

    for tag in ALPHA_TAGS:
        df = load(dataset, tag)
        for m in METHODS:
            acc_final[m].append(df[f"{m}_acc"].iloc[-1] * 100)
            fair_final[m].append(df[f"{m}_fairness"].iloc[-1])

    x     = np.arange(len(ALPHAS))
    width = 0.18
    alpha_labels = [f"α={a}" for a in ALPHAS]

    for metric_data, ylabel, title, fname, fmt, ystart in [
        (acc_final,  "Final Accuracy (%)",
         f"Final Test Accuracy — {ds_label}",
         f"final_accuracy_bar{'_'+dataset if dataset!='cifar10' else ''}.png",
         "{:.1f}", 0.0),
        (fair_final, "GT-Fairness (Jain's J)",
         f"Final GT-Fairness — {ds_label}",
         f"final_fairness_bar{'_'+dataset if dataset!='cifar10' else ''}.png",
         "{:.3f}", 0.5),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        all_vals = [v for vals in metric_data.values() for v in vals]
        ymax = max(all_vals)
        headroom = (ymax - ystart) * 0.18

        for i, m in enumerate(METHODS):
            bars = ax.bar(x + (i - 1.5) * width, metric_data[m], width,
                          label=LABELS[m], color=COLORS[m], alpha=0.85,
                          edgecolor="white")
            if m == "ours":
                for bar in bars:
                    bar.set_edgecolor("#1a7a1a")
                    bar.set_linewidth(1.5)
            for bar, val in zip(bars, metric_data[m]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (ymax - ystart) * 0.008,
                        fmt.format(val),
                        ha="center", va="bottom",
                        fontsize=6.5, rotation=90,
                        color=COLORS[m],
                        fontweight="bold" if m == "ours" else "normal")

        ax.set_xticks(x)
        ax.set_xticklabels(alpha_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(ystart, ymax + headroom)
        path = os.path.join(OUT_DIR, fname)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Fairness gap bar ─────────────────────────────────────────────────────────

def plot_fairness_gap(dataset="cifar10"):
    ds_label = "CIFAR-10" if dataset == "cifar10" else "FashionMNIST"
    gap_data = {m: [] for m in METHODS if m != "fedavg"}

    for tag in ALPHA_TAGS:
        df = load(dataset, tag)
        baseline = df["fedavg_fairness"].mean()
        for m in [m for m in METHODS if m != "fedavg"]:
            gap_data[m].append(df[f"{m}_fairness"].mean() - baseline)

    x      = np.arange(len(ALPHAS))
    width  = 0.22
    others = [m for m in METHODS if m != "fedavg"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhline(0, color="#999999", linewidth=1.2, linestyle="--", alpha=0.6,
               label="FedAVG baseline (Δ=0)")

    all_gaps = [v for vals in gap_data.values() for v in vals]
    gmin, gmax = min(all_gaps), max(all_gaps)
    grange = (gmax - gmin) if gmax > gmin else 0.01

    for i, m in enumerate(others):
        bars = ax.bar(x + (i - 1) * width, gap_data[m], width,
                      label=LABELS[m], color=COLORS[m], alpha=0.85,
                      edgecolor="white")
        for bar, val in zip(bars, gap_data[m]):
            if val >= 0:
                ypos = bar.get_height() + grange * 0.02
                va = "bottom"
            else:
                ypos = bar.get_height() - grange * 0.02
                va = "top"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.3f}",
                    ha="center", va=va,
                    fontsize=6.5, rotation=90,
                    color=COLORS[m],
                    fontweight="bold" if m == "ours" else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([f"α={a}" for a in ALPHAS])
    ax.set_ylabel("GT-Fairness Δ vs FedAVG")
    ax.set_title(f"GT-Fairness Gap relative to FedAVG — {ds_label}\n"
                 "(positive = better than FedAVG)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(gmin - grange * 0.35, gmax + grange * 0.35)
    suffix = f"_{dataset}" if dataset != "cifar10" else ""
    path = os.path.join(OUT_DIR, f"fairness_gap_bar{suffix}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Comparison table heatmap (both datasets) ─────────────────────────────────

def plot_comparison_table():
    datasets  = ["cifar10", "fashionmnist"]
    ds_labels = ["CIFAR-10", "FashionMNIST"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for row, (ds, ds_label) in enumerate(zip(datasets, ds_labels)):
        acc_mat  = np.zeros((len(METHODS), len(ALPHAS)))
        fair_mat = np.zeros((len(METHODS), len(ALPHAS)))

        for j, tag in enumerate(ALPHA_TAGS):
            df = load(ds, tag)
            for i, m in enumerate(METHODS):
                acc_mat[i, j]  = df[f"{m}_acc"].iloc[-1] * 100
                fair_mat[i, j] = df[f"{m}_fairness"].iloc[-1]

        for col, (mat, title, cmap, vmin, vmax, fmt) in enumerate([
            (acc_mat,  f"{ds_label} — Final Accuracy (%)",
             "YlGn", acc_mat.min() - 1, acc_mat.max() + 0.5, "{:.1f}"),
            (fair_mat, f"{ds_label} — GT-Fairness (Jain's J)",
             "RdYlGn", 0.5, 1.0, "{:.3f}"),
        ]):
            ax = axes[row][col]
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(ALPHAS)))
            ax.set_xticklabels([f"α={a}" for a in ALPHAS], fontsize=9)
            ax.set_yticks(range(len(METHODS)))
            ax.set_yticklabels([LABELS[m] for m in METHODS], fontsize=9)
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.85)

            # find best per column
            best_rows = mat.argmax(axis=0) if col == 0 else mat.argmax(axis=0)

            for i in range(len(METHODS)):
                for j in range(len(ALPHAS)):
                    val = mat[i, j]
                    is_best = (i == best_rows[j])
                    txt_color = "white" if (col == 1 and val < 0.72) else "black"
                    weight = "bold" if is_best else "normal"
                    ax.text(j, i, fmt.format(val),
                            ha="center", va="center",
                            fontsize=9, color=txt_color, fontweight=weight)

    fig.suptitle("FairPoC vs Baselines — Final Results  (GradNorm PoC, 50 rounds, seed=42)\n"
                 "Bold = best per column",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "comparison_table.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Data distribution heatmap (unchanged — dataset not modified) ─────────────

def plot_distribution_heatmap(alpha=0.1):
    try:
        from torchvision import datasets, transforms
        from data.cifar10_loader import dirichlet_partition
    except ImportError:
        print("  Skipping distribution heatmap (torchvision not available)")
        return

    CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                       "dog","frog","horse","ship","truck"]
    print(f"  Loading CIFAR-10 for distribution heatmap (α={alpha}) ...")
    tf = transforms.ToTensor()
    ds = datasets.CIFAR10(os.path.join(BASE, "data", "cache"),
                          train=True, download=False, transform=tf)
    partitions = dirichlet_partition(ds, 10, alpha, seed=42)
    targets    = np.array(ds.targets)

    matrix = np.zeros((10, 10))
    for i, idxs in enumerate(partitions):
        labels = targets[idxs]
        for c in range(10):
            matrix[i, c] = (labels == c).sum() / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=matrix.max())
    plt.colorbar(im, ax=ax, label="Fraction of local data")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Client {i}" for i in range(10)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_title(f"Data Distribution per Client — Dirichlet α={alpha}  (CIFAR-10)")
    for i in range(10):
        for j in range(10):
            val = matrix[i, j]
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.35 else "black")

    path = os.path.join(OUT_DIR, f"data_distribution_alpha{str(alpha).replace('.','')}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(BASE)
    print("\n── Per-alpha convergence curves ──")
    plot_convergence_per_alpha("cifar10")
    plot_convergence_per_alpha("fashionmnist")

    print("\n── Per-alpha GT-fairness curves ──")
    plot_fairness_per_alpha("cifar10")
    plot_fairness_per_alpha("fashionmnist")

    print("\n── Per-alpha GT-fairness bar charts ──")
    plot_fairness_bar_per_alpha("cifar10")
    plot_fairness_bar_per_alpha("fashionmnist")

    print("\n── Final accuracy & fairness bars ──")
    plot_final_bars("cifar10")
    plot_final_bars("fashionmnist")

    print("\n── Fairness gap bars ──")
    plot_fairness_gap("cifar10")
    plot_fairness_gap("fashionmnist")

    print("\n── Comparison table heatmap (both datasets) ──")
    plot_comparison_table()

    print("\n── Data distribution heatmap ──")
    plot_distribution_heatmap(alpha=0.1)

    print("\nAll plots updated.")
