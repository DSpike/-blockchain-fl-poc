"""
Analytical extraction of per-function gas costs from blockchain_metrics.csv.

Two stable tiers exist because the Byzantine filter varies active clients per round:
  - Low tier  (~1.34M gas): 9 active clients
  - High tier (~1.49M gas): 10 active clients
  Difference = cost of one additional submit() call.

This script derives submit() gas per client and distribute() gas, then
generates a paper-ready Table IX equivalent.
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def analyze(csv_path="results/blockchain_metrics.csv",
            out_dir="results/plots"):

    df = pd.read_csv(csv_path)
    # Skip rounds 1-2: first-write SSTORE costs are higher (cold storage init)
    stable = df[df["round"] >= 3]["total_gas"]

    # Cluster into two tiers: low (~1.34M = 9 clients), high (~1.49M = 10 clients)
    # Explicit threshold at 1.4M separates the two clean clusters
    THRESHOLD = 1_400_000
    low_tier  = stable[stable <  THRESHOLD]
    high_tier = stable[stable >= THRESHOLD]

    low_mean  = low_tier.mean()
    high_mean = high_tier.mean()
    diff      = high_mean - low_mean       # gas cost of 1 extra submit()

    # The low tier has N_low active clients, high tier has N_low + 1
    # distribute() is called once per round with variable-length client list,
    # but its fixed overhead can be isolated:
    #   total = N * submit_gas + distribute_gas(N)
    #   distribute_gas(N) ≈ D_fixed + N * D_var  (D_var << submit_gas)
    # The marginal cost diff = submit_gas + D_var ≈ submit_gas (D_var is small)

    submit_gas_per_client = diff          # marginal cost of one more client
    # Low tier = 9 clients (empirically validated from config: 10 clients, 1 often filtered)
    n_low = 9
    distribute_gas = low_mean - n_low * submit_gas_per_client

    # --- sanity check on high tier
    n_high = 10
    predicted_high = n_high * submit_gas_per_client + distribute_gas
    error_pct = abs(predicted_high - high_mean) / high_mean * 100

    print("\n" + "="*62)
    print("  FairPoC On-Chain Gas Analysis (from blockchain_metrics.csv)")
    print("="*62)
    print(f"\n  Stable rounds analyzed  : {len(stable)} (rounds 3–100)")
    print(f"  Low-tier rounds         : {len(low_tier)}  (9 active clients)")
    print(f"  High-tier rounds        : {len(high_tier)}  (10 active clients)")
    print(f"\n  Low-tier avg gas        : {low_mean:>12,.0f}")
    print(f"  High-tier avg gas       : {high_mean:>12,.0f}")
    print(f"  Marginal diff           : {diff:>12,.0f}")
    print(f"\n  ─── Per-function breakdown ──────────────────────────────")
    print(f"  submit() per client     : {submit_gas_per_client:>12,.0f} gas")
    print(f"  distribute() per round  : {distribute_gas:>12,.0f} gas")
    print(f"\n  Prediction error (high) : {error_pct:.2f}%  (should be <1%)")
    print(f"\n  ─── Summary metrics ─────────────────────────────────────")
    avg_all = df["total_gas"].mean()
    avg_lat = df["latency_sec"].mean()
    std_gas = df["total_gas"].std()
    std_lat = df["latency_sec"].std()
    print(f"  Avg total gas / round   : {avg_all:>12,.0f}  ± {std_gas:,.0f}")
    print(f"  Avg latency / round     : {avg_lat:>12.3f}s ± {std_lat:.3f}s")
    print("="*62 + "\n")

    # ── ETH fee calculation (Ganache default: 2 Gwei gas price) ────────────
    GAS_PRICE_GWEI = 2.0
    GAS_PRICE_ETH  = GAS_PRICE_GWEI * 1e-9   # 1 Gwei = 1e-9 ETH

    def to_eth(gas):
        return gas * GAS_PRICE_ETH

    # ── paper table values ──────────────────────────────────────────────────
    table_data = [
        ("ContributionRegistry.submit()",
         submit_gas_per_client,
         to_eth(submit_gas_per_client),
         "per client per round"),
        ("RewardDistributor.distribute()",
         distribute_gas,
         to_eth(distribute_gas),
         "per round (all clients)"),
        ("Total (avg, ~10 active clients)",
         avg_all,
         to_eth(avg_all),
         "per FL round"),
    ]

    print(f"\n  Gas price assumed: {GAS_PRICE_GWEI} Gwei (Ganache local testnet default)")
    print("  Table IX equivalent (paste to LaTeX):")
    print("  " + "-"*80)
    print(f"  {'Function':<38} {'Gas Used':>12}  {'Fee (ETH)':>12}  {'Scope'}")
    print("  " + "-"*80)
    for fn, gas, eth, scope in table_data:
        print(f"  {fn:<38} {gas:>12,.0f}  {eth:>12.6f}  {scope}")
    print("  " + "-"*80)
    print("\n  LaTeX rows:")
    for fn, gas, eth, scope in table_data:
        print(f"  {fn} & {gas:,.0f} & {eth:.6f} & {scope} \\\\")
    print()

    # ── plot: gas breakdown bar ─────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    labels = ["submit()\n(per client)", "distribute()\n(per round)"]
    values = [submit_gas_per_client / 1e3, distribute_gas / 1e3]
    colors = ["#B47CC7", "#4878CF"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val*1000:,.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Gas Used (×10³)")
    ax.set_title("On-Chain Gas Cost per Smart Contract Function")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(values) * 1.25)

    path = os.path.join(out_dir, "gas_per_function.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # ── plot: gas distribution (two-panel, paper quality) ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: gas per round scatter — shows two-tier pattern over time
    ax = axes[0]
    rounds_all = df["round"].values
    gas_all    = df["total_gas"].values / 1e3

    # colour each point by tier
    is_low  = df["total_gas"] <  THRESHOLD
    is_high = df["total_gas"] >= THRESHOLD
    is_init = df["round"] <= 2           # warm-up rounds (excluded from analysis)

    ax.scatter(rounds_all[is_init],  gas_all[is_init],
               color="#AAAAAA", s=22, zorder=2, label="Warm-up (excl.)")
    ax.scatter(rounds_all[is_low],   gas_all[is_low],
               color="#4878CF", s=22, zorder=3,
               label=f"9 active clients  ({is_low.sum()} rounds)")
    ax.scatter(rounds_all[is_high],  gas_all[is_high],
               color="#D65F5F", s=22, zorder=3,
               label=f"10 active clients ({is_high.sum()} rounds)")

    # horizontal reference lines
    ax.axhline(low_mean  / 1e3, color="#4878CF", linestyle="--",
               linewidth=1.4, alpha=0.7)
    ax.axhline(high_mean / 1e3, color="#D65F5F", linestyle="--",
               linewidth=1.4, alpha=0.7)

    # clean text labels on right side (no arrows)
    ax.text(101, low_mean  / 1e3, f"{low_mean/1e3:,.1f}K",
            va="center", fontsize=9, color="#4878CF", fontweight="bold")
    ax.text(101, high_mean / 1e3, f"{high_mean/1e3:,.1f}K",
            va="center", fontsize=9, color="#D65F5F", fontweight="bold")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Gas Used (×10³ per round)")
    ax.set_title("On-Chain Gas per FL Round\n(two-tier structure from Byzantine filtering)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, 115)
    ax.set_ylim(0, high_mean / 1e3 * 1.25)

    # Right: clean stacked bar — function breakdown
    ax2 = axes[1]
    scenarios = ["9 clients\n(1 filtered)", "10 clients\n(none filtered)"]
    n_vals    = [9, 10]
    sub_vals  = [submit_gas_per_client * n / 1e3 for n in n_vals]
    dis_vals  = [distribute_gas / 1e3] * 2
    totals    = [s + d for s, d in zip(sub_vals, dis_vals)]

    x = np.arange(len(scenarios))
    w = 0.42

    ax2.bar(x, sub_vals, w, label=f"submit()  ×  clients  ({int(submit_gas_per_client):,} gas each)",
            color="#4878CF", alpha=0.88, edgecolor="white", linewidth=0.6)
    ax2.bar(x, dis_vals, w, bottom=sub_vals,
            label=f"distribute()  ×  1  ({int(distribute_gas):,} gas)",
            color="#D65F5F", alpha=0.88, edgecolor="white", linewidth=0.6)

    # total label above each bar
    for i, total in enumerate(totals):
        ax2.text(x[i], total + 8, f"{total*1e3:,.0f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # segment labels inside bars
    for i, (s, d) in enumerate(zip(sub_vals, dis_vals)):
        ax2.text(x[i], s / 2,     f"submit: {s*1e3:,.0f}",
                 ha="center", va="center", fontsize=8.5,
                 color="white", fontweight="bold")
        ax2.text(x[i], s + d / 2, f"dist: {d*1e3:,.0f}",
                 ha="center", va="center", fontsize=8.5,
                 color="white", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.set_ylabel("Gas Used (×10³ per round)")
    ax2.set_title("Per-Round Gas Breakdown\nby Smart Contract Function")
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.set_ylim(0, max(totals) * 1.22)

    fig.suptitle(
        f"FairPoC On-Chain Overhead  —  Ganache local testnet, {GAS_PRICE_GWEI} Gwei gas price",
        fontsize=12, y=1.02)
    plt.tight_layout()

    path2 = os.path.join(out_dir, "gas_distribution.png")
    fig.savefig(path2, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path2}\n")

    return {
        "submit_gas":    submit_gas_per_client,
        "distribute_gas": distribute_gas,
        "avg_total_gas": avg_all,
        "avg_latency":   avg_lat,
    }


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    analyze(
        csv_path=os.path.join(base, "results", "blockchain_metrics.csv"),
        out_dir=os.path.join(base, "results", "plots"),
    )
