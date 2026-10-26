"""
Generates two paper-ready tables:
  1. Feature comparison table (Table I equivalent) — FairPoC vs. related work
  2. Simulation parameters table (Table V equivalent)

Outputs: PNG figures + LaTeX rows printed to console.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

CHECK = r"$\checkmark$"
CROSS = r"$\times$"


# ── 1. FEATURE COMPARISON TABLE ──────────────────────────────────────────────

def build_feature_table(out_dir):
    """Table I equivalent: feature checklist across related methods."""

    methods = [
        "FedAVG\n[McMahan 17]",
        "FedDAR\n[Springer 25]",
        "PoCL\n[arXiv 24]",
        "BLADE-FL\n[IEEE 23]",
        "Zainudin\n[IoT J. 24]",
        "FairPoC\n(Ours)",
    ]

    features = [
        "Blockchain\nenforcement",
        "Non-IID\naware",
        "Gradient\ndivergence\nscoring",
        "On-chain PoC\n(no trusted\nserver)",
        "Byzantine\nresilience",
        "Jain's\nfairness\nguarantee",
        "ERC-20\nincentive\nmechanism",
        "Smart contract\nenforced\naggregation",
    ]

    # Rows = methods, Cols = features
    # 1 = yes (check), 0 = no (cross), 0.5 = partial
    data = [
        # BC   nonIID  grad   onchain  byz    jain   erc20  sc
        [0,    0,      0,     0,       0,     0,     0,     0],   # FedAVG
        [0,    1,      1,     0,       0,     0,     0,     0],   # FedDAR
        [1,    0,      0,     0.5,     0,     0,     0,     0.5], # PoCL
        [1,    0,      0,     0.5,     0.5,   0,     0,     0.5], # BLADE-FL
        [1,    0,      0,     0.5,     0,     0,     1,     1],   # Zainudin
        [1,    1,      1,     1,       1,     1,     1,     1],   # Ours
    ]

    mat = np.array(data)   # shape (n_methods, n_features)

    n_m = len(methods)
    n_f = len(features)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-0.5, n_f - 0.5)
    ax.set_ylim(-0.5, n_m - 0.5)

    # color map: 0=red, 0.5=orange, 1=green
    cell_colors = {0: "#FFCCCC", 0.5: "#FFE4B5", 1.0: "#CCFFCC"}
    sym_colors  = {0: "#CC0000", 0.5: "#CC7700", 1.0: "#007700"}
    symbols     = {0: CROSS, 0.5: r"$\sim$", 1.0: CHECK}

    for i, method_row in enumerate(data):
        for j, val in enumerate(method_row):
            bg  = cell_colors[val]
            sym = symbols[val]
            sc  = sym_colors[val]
            row_idx = n_m - 1 - i   # flip so Ours is at top visually

            ax.add_patch(plt.Rectangle(
                (j - 0.5, row_idx - 0.5), 1, 1,
                facecolor=bg, edgecolor="white", linewidth=1.5
            ))
            ax.text(j, row_idx, sym, ha="center", va="center",
                    fontsize=14, color=sc,
                    fontweight="bold" if i == n_m - 1 else "normal")

    # highlight "Ours" row
    ours_row = 0   # after flip, ours is at index 0
    ax.add_patch(plt.Rectangle(
        (-0.5, ours_row - 0.5), n_f, 1,
        facecolor="none", edgecolor="#B47CC7", linewidth=2.5, zorder=5
    ))

    # axes labels
    ax.set_xticks(range(n_f))
    ax.set_xticklabels(features, fontsize=8.5, ha="center")
    ax.xaxis.set_tick_params(length=0)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.set_yticks(range(n_m))
    ax.set_yticklabels(list(reversed(methods)), fontsize=9)
    ax.yaxis.set_tick_params(length=0)

    ax.set_title("Table I: Comparison of Blockchain-FL Methods for Contribution Fairness",
                 fontsize=11, pad=16)

    # legend
    patches = [
        mpatches.Patch(color="#CCFFCC", label=r"$\checkmark$  Supported"),
        mpatches.Patch(color="#FFE4B5", label=r"$\sim$  Partial"),
        mpatches.Patch(color="#FFCCCC", label=r"$\times$  Not supported"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              bbox_to_anchor=(1.0, -0.02))

    path = os.path.join(out_dir, "table1_feature_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # LaTeX output
    print("\n  LaTeX (feature comparison table):")
    print("  " + "-"*80)
    col_heads = " & ".join(f.replace("\n", " ") for f in features)
    print(f"  Method & {col_heads} \\\\")
    print("  \\hline")
    sym_tex = {0: r"$\times$", 0.5: r"$\sim$", 1.0: r"$\checkmark$"}
    for method, row in zip(methods, data):
        name = method.replace("\n", " ")
        bold = "Ours" in name
        cells = " & ".join(sym_tex[v] for v in row)
        line = f"  \\textbf{{{name}}} & {cells} \\\\" if bold else f"  {name} & {cells} \\\\"
        print(line)
    print("  " + "-"*80)


# ── 2. PARAMETERS TABLE ───────────────────────────────────────────────────────

def build_params_table(out_dir):
    """Table V equivalent: simulation hyperparameter configuration."""

    params = [
        # (Category, Parameter, Value, Description)
        ("Federated Learning",  "Number of clients",         "10",                  "Edge devices participating in FL"),
        ("Federated Learning",  "Communication rounds",      "100",                 "Total FL training rounds"),
        ("Federated Learning",  "Local epochs",              "2",                   "Local training epochs per round"),
        ("Federated Learning",  "Batch size",                "128",                 "Local mini-batch size"),
        ("Federated Learning",  "Learning rate",             "0.01",                "SGD optimizer, fixed schedule"),
        ("Federated Learning",  "Optimizer",                 "SGD",                 "Stochastic gradient descent"),
        ("Federated Learning",  "Model",                     "ResNet-8",            "8-layer residual CNN"),
        ("Non-IID Setting",     "Dirichlet α",               "0.1, 0.5, 1.0, 10.0","Data heterogeneity sweep"),
        ("Non-IID Setting",     "Dataset",                   "CIFAR-10",            "10-class image classification"),
        ("PoC Scoring",         "Accuracy weight (α)",       "0.5",                 "Weight for validation accuracy term"),
        ("PoC Scoring",         "Divergence weight (β)",     "0.5",                 "Weight for gradient divergence term"),
        ("PoC Scoring",         "Byzantine threshold",       "μ − 2σ",              "Filter clients below mean−2×std"),
        ("Blockchain",          "Platform",                  "Ganache v7.9.2",      "Local Ethereum testnet"),
        ("Blockchain",          "Gas limit",                 "3,000,000",           "Per-transaction gas cap"),
        ("Blockchain",          "Gas price",                 "2 Gwei",              "Ganache default"),
        ("Blockchain",          "Consensus threshold",       "67%",                 "Minimum miner agreement"),
        ("Blockchain",          "Miners",                    "2",                   "Voting nodes for consensus"),
        ("Blockchain",          "Reward pool / round",       "100 tokens",          "ERC-20 tokens distributed per round"),
        ("Reproducibility",     "Random seed",               "42",                  "PyTorch + NumPy fixed seed"),
        ("Reproducibility",     "Statistical runs",          "5",                   "Independent seeds for mean ± std"),
    ]

    categories = []
    seen = set()
    for cat, *_ in params:
        if cat not in seen:
            categories.append(cat)
            seen.add(cat)

    # assign row colors by category
    cat_colors = {
        "Federated Learning": "#EAF4FB",
        "Non-IID Setting":    "#FEF9E7",
        "PoC Scoring":        "#EAFAF1",
        "Blockchain":         "#F9EBEA",
        "Reproducibility":    "#F5EEF8",
    }

    n = len(params)
    fig, ax = plt.subplots(figsize=(13, n * 0.38 + 0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")

    col_x    = [0.0, 0.22, 0.42, 0.55, 1.0]
    headers  = ["Category", "Parameter", "Value", "Description"]

    # header row
    for hx, hname in zip(col_x, headers):
        ax.text(hx + 0.01, n - 0.3, hname,
                fontsize=9, fontweight="bold", va="center",
                transform=ax.transData)
    ax.axhline(n - 0.5, color="black", linewidth=1.2)

    for idx, (cat, param, val, desc) in enumerate(reversed(params)):
        y    = idx + 0.5
        bg   = cat_colors.get(cat, "#FFFFFF")
        ax.add_patch(plt.Rectangle((0, idx), 1, 1,
                                   facecolor=bg, edgecolor="white",
                                   linewidth=0.5, transform=ax.transData))
        row_data = [cat, param, val, desc]
        for xi, (x0, x1, txt) in enumerate(zip(col_x, col_x[1:], row_data)):
            ax.text(x0 + 0.01, y, txt,
                    fontsize=8.5, va="center", transform=ax.transData,
                    fontweight="bold" if xi == 1 else "normal")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Table V: Simulation Parameter Configuration",
                 fontsize=11, pad=10)

    path = os.path.join(out_dir, "table5_parameters.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # LaTeX output
    print("\n  LaTeX (parameters table):")
    print("  " + "-"*80)
    print("  Category & Parameter & Value & Description \\\\")
    print("  \\hline")
    prev_cat = None
    for cat, param, val, desc in params:
        cat_cell = cat if cat != prev_cat else ""
        print(f"  {cat_cell} & {param} & {val} & {desc} \\\\")
        prev_cat = cat
    print("  " + "-"*80)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main(out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    print("\n[1/2] Feature comparison table ...")
    build_feature_table(out_dir)
    print("\n[2/2] Parameters table ...")
    build_params_table(out_dir)
    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    main(out_dir=os.path.join(base, "results", "plots"))
