"""
Read aggregated multi-seed CSVs (comparison_ms_alpha*.csv) and print
ready-to-paste LaTeX table rows for main.tex Tables I and II.

Format: "acc_mean ± acc_std / fair_mean ± fair_std"

Usage (after run_multiseed_comparison.py finishes):
    cd blockchain-fl-poc
    python evaluation/update_paper_tables.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

METHODS_DISPLAY = {
    "fedavg":       "FedAVG",
    "valscore":     "PoCL",
    "cosinefilter": "BLADE-FL",
    "ours":         "FairPoC",
}
ALPHAS   = [0.1, 0.5, 1.0, 10.0]
DATASETS = ["cifar10", "fashionmnist"]
METHODS  = list(METHODS_DISPLAY.keys())


def alpha_tag(a):
    return str(a).replace(".", "")


def ds_tag(ds):
    return f"_{ds}" if ds != "cifar10" else ""


def last_round_stats(csv_path):
    """Read aggregated CSV, return dict method -> (acc_mean, acc_std, fair_mean, fair_std)."""
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]   # last round = final performance

    stats = {}
    for m in METHODS:
        acc_col  = f"{m}_acc_mean"
        acc_scol = f"{m}_acc_std"
        fair_col  = f"{m}_fair_mean"
        fair_scol = f"{m}_fair_std"
        if acc_col in df.columns:
            stats[m] = (last[acc_col] * 100, last[acc_scol] * 100,
                        last[fair_col],       last[fair_scol])
    return stats


def cell(acc_mean, acc_std, fair_mean, fair_std, bold=False):
    a = f"{acc_mean:.1f}$\\pm${acc_std:.1f}"
    f = f"{fair_mean:.3f}$\\pm${fair_std:.3f}"
    s = f"{a}/{f}"
    if bold:
        return f"\\textbf{{{s}}}"
    return s


def print_table(dataset):
    dtag  = ds_tag(dataset)
    title = "CIFAR-10" if dataset == "cifar10" else "FashionMNIST"
    print(f"\n{'='*80}")
    print(f"  TABLE: {title}  (Final Acc% ± std / Avg GT-Fairness ± std)")
    print(f"{'='*80}")

    cols = {a: {} for a in ALPHAS}
    missing = []
    for alpha in ALPHAS:
        tag      = alpha_tag(alpha)
        path     = f"results/comparison_ms{dtag}_alpha{tag}.csv"
        if not os.path.exists(path):
            missing.append(path)
            continue
        cols[alpha] = last_round_stats(path)

    if missing:
        print(f"  MISSING CSVs (run run_multiseed_comparison.py first):")
        for p in missing:
            print(f"    {p}")
        if len(missing) == len(ALPHAS):
            return

    # find best per column (method that wins on acc OR fair — print both)
    best_acc  = {}
    best_fair = {}
    for alpha in ALPHAS:
        if not cols[alpha]:
            continue
        best_acc[alpha]  = max(cols[alpha], key=lambda m: cols[alpha][m][0])
        best_fair[alpha] = max(cols[alpha], key=lambda m: cols[alpha][m][2])

    print(f"\n  Method       | " + " | ".join(f"alpha={a:4}" for a in ALPHAS))
    print(f"  {'─'*14}|" + "─" * 65)

    for m in METHODS:
        name    = METHODS_DISPLAY[m]
        row_str = f"  {name:<12} |"
        for alpha in ALPHAS:
            if not cols[alpha] or m not in cols[alpha]:
                row_str += "  N/A |"
                continue
            am, as_, fm, fs = cols[alpha][m]
            is_best = (m == best_acc.get(alpha) or m == best_fair.get(alpha))
            row_str += " " + cell(am, as_, fm, fs, bold=is_best) + " |"
        print(row_str)

    print(f"\n  LaTeX rows:")
    for m in METHODS:
        name    = METHODS_DISPLAY[m]
        cells   = []
        for alpha in ALPHAS:
            if not cols[alpha] or m not in cols[alpha]:
                cells.append("N/A")
                continue
            am, as_, fm, fs = cols[alpha][m]
            is_best = (m == best_acc.get(alpha) or m == best_fair.get(alpha))
            cells.append(cell(am, as_, fm, fs, bold=is_best))
        print(f"  {name} & " + " & ".join(cells) + r" \\")


def main():
    for ds in DATASETS:
        print_table(ds)
    print("\n\nDone. Paste the LaTeX rows into main.tex Tables I and II.")
    print("Note: bold marks best accuracy OR best GT-Fairness per column.")


if __name__ == "__main__":
    main()
