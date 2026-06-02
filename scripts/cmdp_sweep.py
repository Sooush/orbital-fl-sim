"""
CMDP Pareto sweep for on-orbit bandwidth allocation.

Sweeps the Lagrange multiplier, solves discounted value iteration at each point,
and evaluates the greedy policy by simulator rollout over many seeds. Every
operating point (baselines and CMDP) carries a 95% confidence interval, and the
headline gain is reported as the comparison the paper actually makes: the CMDP at
zero energy-floor violations against the best zero-violation heuristic.

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
    SatelliteEnv, ANALYSIS_CFG,
    power_oblivious, make_power_weighted, contact_gated,
)
from orbital_fl.cmdp import CMDPConfig, pareto_sweep

CMDP_CFG = CMDPConfig(Q_max=20, gamma=0.99, tol=0.01, max_iter=1000)
LAM_VALUES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0, 25.0]
HORIZON = 5000
SEEDS = range(20)


def baseline_ci(policy_fn, cfg, horizon, seeds):
    """Multi-seed throughput and energy-floor cost for a heuristic policy."""
    seeds = list(seeds)
    thr = np.empty(len(seeds)); cost = np.empty(len(seeds))
    for k, sd in enumerate(seeds):
        env = SatelliteEnv(config=cfg)
        obs = env.reset(seed=sd)
        tr = tc = 0.0
        for _ in range(horizon):
            obs, r, _, info = env.step(policy_fn(obs))
            tr += r; tc += info["cost"]
        thr[k] = tr / horizon; cost[k] = tc / horizon
    def ci(x):
        return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return {"thr": float(thr.mean()), "thr_ci": float(ci(thr)),
            "cost": float(cost.mean()), "cost_ci": float(ci(cost)),
            "cost_max": float(cost.max())}


def main():
    cfg = ANALYSIS_CFG

    print(f"Evaluating baseline policies ({len(list(SEEDS))} seeds)...")
    baselines = {
        "power_oblivious": baseline_ci(power_oblivious, cfg, HORIZON, SEEDS),
        "contact_gated":   baseline_ci(contact_gated, cfg, HORIZON, SEEDS),
        "power_weighted":  baseline_ci(make_power_weighted(cfg.B_crit, cfg.B_max), cfg, HORIZON, SEEDS),
    }
    for name, b in baselines.items():
        print(f"  {name:16s}  thr={b['thr']:.3f}+/-{b['thr_ci']:.3f}  "
              f"cost={b['cost']:.3f}+/-{b['cost_ci']:.3f}")

    print()
    pareto = pareto_sweep(cfg, CMDP_CFG, LAM_VALUES, HORIZON, seeds=SEEDS)
    # columns: lam, thr_mean, thr_ci, cost_mean, cost_ci

    # --- figure --- #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(pareto[:, 3], pareto[:, 1], xerr=pareto[:, 4], yerr=pareto[:, 2],
                fmt="o-", color="steelblue", linewidth=2, markersize=5, capsize=2,
                label="CMDP Pareto frontier", zorder=3)
    for idx, lam in enumerate(LAM_VALUES):
        if lam in {0.0, 0.8, 6.0, 25.0}:
            ax.annotate(f"$\\theta={lam}$", xy=(pareto[idx, 3], pareto[idx, 1]),
                        xytext=(pareto[idx, 3] + 0.006, pareto[idx, 1] - 0.01),
                        fontsize=7.5, color="steelblue")

    markers = ["s", "^", "D"]; colors = ["tomato", "forestgreen", "darkorange"]
    bl_labels = {"power_oblivious": "power\\_oblivious",
                 "contact_gated": "contact\\_gated",
                 "power_weighted": "power\\_weighted"}
    for (name, b), m, col in zip(baselines.items(), markers, colors):
        ax.errorbar(b["cost"], b["thr"], xerr=b["cost_ci"], yerr=b["thr_ci"],
                    fmt=m, markersize=9, color=col, zorder=5, capsize=2,
                    markeredgecolor="black", markeredgewidth=0.5, label=bl_labels[name])

    ax.set_xlabel("Energy-floor fraction  (cost $= \\mathbf{1}[B < B_{\\mathrm{crit}}]$)")
    ax.set_ylabel("Mean throughput  (gradients / slot)")
    ax.set_title("CMDP Pareto frontier vs heuristic baselines (20 seeds, 95% CI)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.abspath(os.path.join(HERE, "..", "figures", "cmdp_pareto.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"\nSaved: {out}")

    # --- summary table --- #
    print("\n--- CMDP Pareto points ---")
    print(f"{'theta':>6}  {'throughput':>18}  {'cost_frac':>18}")
    for lam, tm, tci, cm, cci in pareto:
        print(f"{lam:6.2f}  {tm:8.3f} +/- {tci:5.3f}   {cm:8.3f} +/- {cci:5.3f}")

    # --- headline comparison: zero-violation CMDP vs best zero-violation heuristic --- #
    zero_idx = [i for i, r in enumerate(pareto) if r[3] <= 1e-3]
    pw = baselines["power_weighted"]
    print("\n--- Headline comparison (the paper's claim) ---")
    if zero_idx:
        # smallest theta achieving (near) zero violations
        i0 = min(zero_idx, key=lambda i: pareto[i, 0])
        cmdp_thr = pareto[i0, 1]
        gain = (cmdp_thr - pw["thr"]) / pw["thr"] * 100
        print(f"  CMDP (theta={pareto[i0,0]:.2f}) thr={cmdp_thr:.3f}  cost={pareto[i0,3]:.3f}")
        print(f"  power_weighted        thr={pw['thr']:.3f}  cost={pw['cost']:.3f} (cost_max {pw['cost_max']:.3f})")
        print(f"  Throughput gain at zero violations: {gain:+.1f}%")
    else:
        print("  No CMDP point reached zero energy-floor violations in this sweep.")


if __name__ == "__main__":
    main()
