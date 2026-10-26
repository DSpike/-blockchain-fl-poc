"""
Table VIII equivalent: per-method, per-alpha final accuracy and avg fairness.
Generates:
  - Console LaTeX rows (copy-paste to paper)
  - results/plots/comparison_table.png (publication-quality figure)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

METHODS = ["fedavg", "pocl", "blade_fl", "ours"]
LABELS  = {
    "fedavg":   "FedAVG",
    "pocl":     "PoCL",
    "blade_fl": "BLADE-FL",
    "ours":     "FairPoC (Ours)",
}
ALPHA_TAG = {"01": 0.1, "05": 0.5, "10": 1.0, "100": 10.0}


def load_results(results_dir):
    """Returns dict: alpha -> {method -> {acc, fairness_avg, fairness_min}}"""
    data = {}
    for fpath in sorted(glob.glob(os.path.join(results_dir, "comparison_alpha*.csv"))):
        tag = os.path.basename(fpath).replace("comparison_alpha", "").replace(".csv", "")
        alpha = ALPHA_TAG.get(tag, float(tag))
        df = pd.read_csv(fpath)
        data[alpha] = {}
        for m in METHODS:
            data[alpha][m] = {
                "acc":          df[f"{m}_acc"].iloc[-1] * 100,
                "fairness_avg": df[f"{m}_fairness"].mean() * 100,
                "fairness_min": df[f"{m}_fairness"].min() * 100,
            }
    return data


def print_latex(data):
    alphas = sorted(data.keys())
    print("\n" + "="*90)
    print("  Comparison Table — Final Accuracy (%) / Avg Fairness (%) per Method × Alpha")
    print("="*90)

    # Header
    header = f"  {'Method':<18}"
    for a in alphas:
        header += f"  α={a:<4}  Acc(%) / Fair(%)"
    print(header)
    print("  " + "-"*88)

    for m in METHODS:
        row = f"  {LABELS[m]:<18}"
        for a in alphas:
            acc  = data[a][m]["acc"]
            fair = data[a][m]["fairness_avg"]
            row += f"  {acc:5.2f} / {fair:6.3f}      "
        print(row)
    print("  " + "-"*88)

    # LaTeX table block
    print("\n  LaTeX rows (multicolumn format):")
    print("  %  Method  & α=0.1 Acc & α=0.1 Fair & α=0.5 Acc & α=0.5 Fair & α=1.0 Acc & α=1.0 Fair & α=10 Acc & α=10 Fair")
    for m in METHODS:
        cells = []
        for a in alphas:
            acc  = data[a][m]["acc"]
            fair = data[a][m]["fairness_avg"]
            cells.append(f"{acc:.2f}")
            cells.append(f"{fair:.3f}")
        bold = m == "ours"
        name = f"\\textbf{{{LABELS[m]}}}" if bold else LABELS[m]
        vals = " & ".join(
            f"\\textbf{{{c}}}" if bold else c for c in cells
        )
        print(f"  {name} & {vals} \\\\")
    print()

    # Fairness gap highlight
    print("  Fairness gap (1.000 - avg_fairness) at α=0.1 (extreme non-IID):")
    for m in METHODS:
        gap = 100.0 - data[0.1][m]["fairness_avg"]
        print(f"    {LABELS[m]:<18}: {gap:.4f}%")
    print("="*90 + "\n")


def plot_table(data, out_dir):
    alphas  = sorted(data.keys())
    n_alpha = len(alphas)
    n_meth  = len(METHODS)

    acc_mat  = np.zeros((n_meth, n_alpha))
    fair_mat = np.zeros((n_meth, n_alpha))

    for j, a in enumerate(alphas):
        for i, m in enumerate(METHODS):
            acc_mat[i, j]  = data[a][m]["acc"]
            fair_mat[i, j] = data[a][m]["fairness_avg"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    xlabels = [f"α={a}" for a in alphas]
    ylabels = [LABELS[m] for m in METHODS]

    for ax, mat, title, fmt, vmin in [
        (axes[0], acc_mat,  "Final Test Accuracy (%)",    ".2f", None),
        (axes[1], fair_mat, "Avg Contribution Fairness (Jain's × 100)", ".3f", 95.0),
    ]:
        im = ax.imshow(mat, cmap="YlGn",
                       vmin=vmin if vmin else mat.min() - 1,
                       vmax=mat.max() + 0.5,
                       aspect="auto")
        ax.set_xticks(range(n_alpha)); ax.set_xticklabels(xlabels, fontsize=11)
        ax.set_yticks(range(n_meth));  ax.set_yticklabels(ylabels, fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        plt.colorbar(im, ax=ax, shrink=0.85)

        # cell values
        for i in range(n_meth):
            for j in range(n_alpha):
                val = mat[i, j]
                # bold if best in column
                col_best = np.argmax(mat[:, j])
                weight = "bold" if i == col_best else "normal"
                color  = "white" if val < (mat.min() + (mat.max()-mat.min())*0.3) else "black"
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center",
                        fontsize=10, fontweight=weight, color=color)

    fig.suptitle("FairPoC vs. Baselines: Accuracy and Fairness Across Heterogeneity Levels",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(out_dir, "comparison_table.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fairness_gap(data, out_dir):
    """Bar chart showing fairness gap (1 - Jain's) at each alpha — key paper figure."""
    alphas = sorted(data.keys())
    x      = np.arange(len(alphas))
    width  = 0.18

    COLORS = {
        "fedavg":   "#4878CF",
        "pocl":     "#D65F5F",
        "blade_fl": "#6ACC65",
        "ours":     "#B47CC7",
    }

    # y-axis ceiling: largest gap among baselines, with room for labels
    max_gap = max(
        100.0 - data[a][m]["fairness_avg"]
        for a in alphas for m in METHODS if m != "ours"
    )
    y_ceil = max_gap * 1.45

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, m in enumerate(METHODS):
        gaps   = [100.0 - data[a][m]["fairness_avg"] for a in alphas]
        offset = (i - len(METHODS) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, gaps, width, label=LABELS[m],
                        color=COLORS[m], alpha=0.85)

        for bar, gap in zip(bars, gaps):
            bx = bar.get_x() + bar.get_width() / 2
            if gap < 0.001:
                # FairPoC zero gap — draw a visible tick at the baseline + label
                ax.plot([bx - width*0.35, bx + width*0.35], [0, 0],
                        color=COLORS[m], linewidth=2.5, solid_capstyle="round")
                ax.text(bx, y_ceil * 0.03, "0.000",
                        ha="center", va="bottom", fontsize=8,
                        color=COLORS[m], fontweight="bold")
            else:
                ax.text(bx, gap + y_ceil * 0.01,
                        f"{gap:.3f}", ha="center", va="bottom",
                        fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"α={a}\n({'Extreme' if a==0.1 else 'High' if a==0.5 else 'Mild' if a==1.0 else 'Near-IID'})"
        for a in alphas
    ])
    ax.set_ylabel("Fairness Gap (100 − Avg Jain's × 100)  [lower is better]")
    ax.set_title("Contribution Fairness Gap vs. Data Heterogeneity")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, y_ceil)

    path = os.path.join(out_dir, "fairness_gap_bar.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main(results_dir="results", out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    data = load_results(results_dir)

    if not data:
        print("No comparison CSVs found.")
        return

    print_latex(data)
    print("[1/2] Heatmap comparison table ...")
    plot_table(data, out_dir)
    print("[2/2] Fairness gap bar chart ...")
    plot_fairness_gap(data, out_dir)
    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    main(
        results_dir=os.path.join(base, "results"),
        out_dir=os.path.join(base, "results", "plots"),
    )
