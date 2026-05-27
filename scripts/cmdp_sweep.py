"""
CMDP Pareto sweep for on-orbit bandwidth allocation.

Sweeps the Lagrange multiplier lam, solves discounted value iteration for
each point, evaluates the resulting policy by simulator rollout, and plots
the throughput vs. energy-floor-fraction Pareto frontier alongside baseline
policy operating points.

Run:  python scripts/cmdp_sweep.py
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

from orbital_fl import (
    SatelliteEnv, EnvConfig,
    power_oblivious, make_power_weighted, contact_gated,
)
from orbital_fl.cmdp import CMDPConfig, pareto_sweep

# Same calibrated config as queueing_analysis.py
ANALYSIS_CFG = EnvConfig(
    B_max=20, B_crit=4, B_init=10,
    eta=4.0,
    P_base=0.5, P_gpu=2.0, P_tx=1.5,
    p_eclipse_exit=0.028, p_eclipse_enter=0.018,
    p_link_acquire=0.10,  p_link_lose=0.05,
    lambda_g=0.5, mu=2, Q_max=50,
    seed=42,
)

CMDP_CFG = CMDPConfig(Q_max=20, gamma=0.99, tol=0.01, max_iter=1000)

LAM_VALUES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0, 25.0]

HORIZON = 5000


def _baseline(policy_fn, cfg, horizon, seed=0):
    env = SatelliteEnv(config=cfg)
    obs = env.reset(seed=seed)
    tr, tc = 0.0, 0.0
    for _ in range(horizon):
        obs, r, _, info = env.step(policy_fn(obs))
        tr += r
        tc += info["cost"]
    return tr / horizon, tc / horizon


def main():
    cfg = ANALYSIS_CFG

    # --- baseline operating points ---
    print("Evaluating baseline policies...")
    baselines = {
        "power_oblivious": _baseline(power_oblivious, cfg, HORIZON),
        "contact_gated":   _baseline(contact_gated,   cfg, HORIZON),
        "power_weighted":  _baseline(
            make_power_weighted(cfg.B_crit, cfg.B_max), cfg, HORIZON),
    }
    for name, (thr, cost) in baselines.items():
        print(f"  {name:20s}  thr={thr:.3f}  cost={cost:.3f}")

    # --- CMDP sweep ---
    print()
    pareto = pareto_sweep(cfg, CMDP_CFG, LAM_VALUES, HORIZON, seed=0)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(pareto[:, 2], pareto[:, 1], "o-", color="steelblue",
            linewidth=2, markersize=5, label="CMDP Pareto frontier", zorder=3)

    for idx, lam in enumerate(LAM_VALUES):
        if lam in {0.0, 0.8, 6.0, 25.0}:
            ax.annotate(
                f"$\\lambda={lam}$",
                xy=(pareto[idx, 2], pareto[idx, 1]),
                xytext=(pareto[idx, 2] + 0.006, pareto[idx, 1] - 0.008),
                fontsize=7.5, color="steelblue",
            )

    markers = ["s", "^", "D"]
    colors  = ["tomato", "forestgreen", "darkorange"]
    bl_labels = {
        "power_oblivious": "power\\_oblivious",
        "contact_gated":   "contact\\_gated",
        "power_weighted":  "power\\_weighted",
    }
    for (name, (thr, cost)), m, col in zip(baselines.items(), markers, colors):
        ax.scatter(cost, thr, marker=m, s=90, color=col, zorder=5,
                   label=bl_labels[name], edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Energy-floor fraction  (cost = $\\mathbf{1}[B < B_{\\mathrm{crit}}]$)")
    ax.set_ylabel("Mean throughput  (gradients / slot)")
    ax.set_title("CMDP Pareto Frontier: Throughput vs. Energy-Floor Frequency")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.abspath(os.path.join(HERE, "..", "figures", "cmdp_pareto.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"\nSaved: {out}")

    # --- summary table ---
    print("\n--- CMDP Pareto Points ---")
    print(f"{'lam':>6}  {'throughput':>10}  {'cost_frac':>10}")
    for lam, thr, cost in pareto:
        print(f"{lam:6.2f}  {thr:10.3f}  {cost:10.3f}")

    print("\n--- Baseline vs CMDP at matched cost ---")
    cg_cost = baselines["contact_gated"][1]
    cg_thr  = baselines["contact_gated"][0]
    # find closest CMDP point in cost
    idx = np.argmin(np.abs(pareto[:, 2] - cg_cost))
    cmdp_thr = pareto[idx, 1]
    if cg_thr > 0:
        gain = (cmdp_thr - cg_thr) / cg_thr * 100
        print(f"  contact_gated thr={cg_thr:.3f}  cost={cg_cost:.3f}")
        print(f"  CMDP (lam={pareto[idx,0]:.2f}) thr={cmdp_thr:.3f}  cost={pareto[idx,2]:.3f}")
        print(f"  Throughput gain at matched cost: {gain:+.1f}%")


if __name__ == "__main__":
    main()
