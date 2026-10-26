"""Generate plots from blockchain FL simulation results."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

BC_COLOR  = "#B47CC7"   # purple — matches "ours" in other plots
FL_COLOR  = "#4878CF"   # blue — matches FedAVG


def plot_blockchain_convergence(bc_conv, fl_conv, out_dir):
    """Blockchain FL vs pure FL-only accuracy and fairness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # accuracy
    ax1.plot(bc_conv["round"], bc_conv["accuracy"] * 100,
             color=BC_COLOR, linewidth=1.8, label="Blockchain FL (Ours)")
    if fl_conv is not None:
        rounds = fl_conv["round"].values
        ax1.plot(rounds, fl_conv["fedavg_acc"].values * 100,
                 color=FL_COLOR, linewidth=1.8, linestyle="--", label="FL-only (FedAVG)")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Convergence: Blockchain FL vs FL-only")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # fairness
    ax2.plot(bc_conv["round"], bc_conv["fairness"],
             color=BC_COLOR, linewidth=1.8, label="Blockchain FL (Ours)")
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.5)
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Fairness Index (Jain's)")
    ax2.set_title("Fairness: Blockchain FL (100 rounds)")
    ax2.set_ylim(0.98, 1.005)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(out_dir, "blockchain_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gas_latency(metrics, out_dir):
    """Gas cost and latency per round."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # gas
    ax1.plot(metrics["round"], metrics["total_gas"] / 1e6,
             color=BC_COLOR, linewidth=1.8)
    ax1.axhline(metrics["total_gas"].mean() / 1e6, color="black",
                linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"Avg: {metrics['total_gas'].mean()/1e6:.2f}M")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Gas Used (millions)")
    ax1.set_title("On-Chain Gas Cost per Round")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # latency
    ax2.plot(metrics["round"], metrics["latency_sec"],
             color=BC_COLOR, linewidth=1.8)
    ax2.axhline(metrics["latency_sec"].mean(), color="black",
                linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"Avg: {metrics['latency_sec'].mean():.3f}s")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("On-Chain Latency (seconds)")
    ax2.set_title("Blockchain Transaction Latency per Round")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(out_dir, "blockchain_overhead.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overhead_bar(metrics, out_dir):
    """Summary bar chart of avg gas and latency — for paper table."""
    avg_gas     = metrics["total_gas"].mean() / 1e6
    avg_latency = metrics["latency_sec"].mean()
    std_gas     = metrics["total_gas"].std() / 1e6
    std_latency = metrics["latency_sec"].std()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].bar(["FairPoC\n(Ours)"], [avg_gas], yerr=[std_gas],
                color=BC_COLOR, alpha=0.85, capsize=6, width=0.4)
    axes[0].set_ylabel("Avg Gas per Round (M)")
    axes[0].set_title("On-Chain Gas Cost")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].text(0, avg_gas + std_gas + 0.01, f"{avg_gas:.2f}M",
                 ha="center", va="bottom", fontsize=10)

    axes[1].bar(["FairPoC\n(Ours)"], [avg_latency], yerr=[std_latency],
                color=BC_COLOR, alpha=0.85, capsize=6, width=0.4)
    axes[1].set_ylabel("Avg Latency per Round (s)")
    axes[1].set_title("On-Chain Transaction Latency")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].text(0, avg_latency + std_latency + 0.002, f"{avg_latency:.3f}s",
                 ha="center", va="bottom", fontsize=10)

    fig.suptitle("Blockchain Overhead per FL Round (100 rounds avg)", y=1.02)
    path = os.path.join(out_dir, "blockchain_overhead_bar.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main(results_dir="results", out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)

    bc_conv = pd.read_csv(f"{results_dir}/blockchain_convergence.csv")
    metrics  = pd.read_csv(f"{results_dir}/blockchain_metrics.csv")

    # load FL-only comparison for overlay (alpha=0.5 matches blockchain config)
    fl_conv = None
    fl_path = f"{results_dir}/comparison_alpha05.csv"
    if os.path.exists(fl_path):
        fl_conv = pd.read_csv(fl_path)

    print("\n[1/3] Convergence plot ...")
    plot_blockchain_convergence(bc_conv, fl_conv, out_dir)

    print("[2/3] Gas + latency per round ...")
    plot_gas_latency(metrics, out_dir)

    print("[3/3] Overhead summary bar ...")
    plot_overhead_bar(metrics, out_dir)

    print(f"\nAll blockchain plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
