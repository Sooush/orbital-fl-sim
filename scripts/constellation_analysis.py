"""
Constellation analysis: product form, merged rate, and correlated-eclipse breakdown.

Tests the two structural claims of the N-satellite section against simulation:
  1. Merged arrival rate at the aggregator equals N*lam (flow balance).
  2. With independent per-satellite chains the joint backlog factors (product
     form): queue correlation and total-variation distance to the product of
     marginals are near zero.
Then drives the breakdown regime by sharing the illumination (eclipse) chain
across the plane with correlation rho_corr in [0,1], showing per-satellite
queues become positively correlated and aggregation delay climbs above the
independent-queue prediction.

Run:  python scripts/constellation_analysis.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from orbital_fl.constellation import simulate_constellation
from orbital_fl.analysis_config import ANALYSIS_CFG

LAM = 0.3                      # per-satellite gradient rate (stable aggregator)
HORIZON = 40000
N_VALUES = [1, 2, 4, 8, 16]
RHO_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_BREAK = 4                    # constellation size for the breakdown sweep


def main():
    cfg = ANALYSIS_CFG

    # --- 1. merged rate vs N (independent chains) --- #
    print("Merged arrival rate vs N (independent chains):")
    merged = []
    for N in N_VALUES:
        r = simulate_constellation(cfg, N, LAM, horizon=HORIZON, seed=0, rho_corr=0.0)
        merged.append(r.merged_rate)
        print(f"  N={N:2d}  merged={r.merged_rate:6.3f}  N*lam={N*LAM:6.3f}  "
              f"per_sat_thr={r.per_sat_throughput:.3f}")

    # --- 2. product form at independence (rho_corr = 0) --- #
    print("\nProduct-form diagnostics at independence (rho_corr=0):")
    r2 = simulate_constellation(cfg, 2, LAM, horizon=HORIZON, seed=1, rho_corr=0.0)
    print(f"  N=2  queue_corr={r2.queue_corr:+.3f}  TV={r2.tv_distance:.3f}  "
          f"(both near 0 confirm product form)")
    tv_baseline = r2.tv_distance

    # --- 3. correlated-eclipse breakdown sweep --- #
    print(f"\nCorrelated-eclipse breakdown (N={N_BREAK}):")
    print(f"  {'rho_corr':>8} {'queue_corr':>11} {'TV':>7} {'agg_delay':>10} {'indep_pred':>11}")
    qcorr, tvs, aggd = [], [], []
    indep_delay = None
    for rho in RHO_VALUES:
        r = simulate_constellation(cfg, N_BREAK, LAM, horizon=HORIZON, seed=2, rho_corr=rho)
        qcorr.append(r.queue_corr); tvs.append(r.tv_distance); aggd.append(r.agg_mean_delay)
        if rho == 0.0:
            indep_delay = r.agg_mean_delay
        print(f"  {rho:8.2f} {r.queue_corr:+11.3f} {r.tv_distance:7.3f} "
              f"{r.agg_mean_delay:10.2f} {r.agg_indep_delay:11.2f}")

    # --- figure --- #
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Constellation: superposition of on/off gradient sources", fontsize=13)

    ax = axes[0, 0]
    ax.plot(N_VALUES, merged, "o-", color="steelblue", label="measured merged rate")
    ax.plot(N_VALUES, [N * LAM for N in N_VALUES], "k--", label="$N\\lambda$ (flow balance)")
    ax.set_xlabel("Constellation size $N$")
    ax.set_ylabel("Aggregator input rate (grad/slot)")
    ax.set_title("(a) Merged arrival rate vs $N$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(RHO_VALUES, qcorr, "o-", color="firebrick")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Eclipse correlation $\\rho_{\\mathrm{corr}}$")
    ax.set_ylabel("Queue correlation $\\mathrm{corr}(Q_1, Q_2)$")
    ax.set_title(f"(b) Product form breaks with correlation ($N={N_BREAK}$)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(RHO_VALUES, aggd, "o-", color="seagreen", label="measured aggregator delay")
    if indep_delay is not None:
        ax.axhline(indep_delay, color="gray", linestyle="--", linewidth=1.2,
                   label="independent baseline ($\\rho_{\\mathrm{corr}}=0$)")
    ax.set_xlabel("Eclipse correlation $\\rho_{\\mathrm{corr}}$")
    ax.set_ylabel("Mean aggregation delay (slots)")
    ax.set_title("(c) Aggregation delay vs eclipse correlation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(RHO_VALUES, tvs, "o-", color="darkorange", label="TV(joint, product)")
    ax.axhline(tv_baseline, color="gray", linestyle=":", linewidth=1,
               label="finite-sample baseline")
    ax.set_xlabel("Eclipse correlation $\\rho_{\\mathrm{corr}}$")
    ax.set_ylabel("Total-variation distance")
    ax.set_title("(d) Factorization error vs correlation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.abspath(os.path.join(HERE, "..", "figures", "constellation.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
