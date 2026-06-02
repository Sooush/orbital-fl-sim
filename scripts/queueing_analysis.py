"""
Queueing analysis for the Federated Orbital Edge Mesh project.

Pipeline:
  1. Build and solve the (B, I, C) energy-channel Markov chain (calibrated cfg).
  2. Extract P_ready and the link-recovery rate beta.
  3. Compare three analytical staleness models against simulation across load:
       (i)   single-geometric vacation       E[T] = 1/(mu-lam) + (2-beta)/(2 beta)
       (ii)  vacation with the true off-period 2nd moment (markov.offperiod_moments)
       (iii) the exact joint (Q, B, I, C) Markov-modulated solve (markov.solve_queue_chain)
     against a Poisson(lam) discrete-event rollout with 95% confidence intervals.
  4. Generate figures/queueing_analysis.png.

The arrival stream in the validation is an independent Poisson(lam) process, which
matches the assumption behind every analytical model. This is the load-matched
fix to the earlier validation, which drove arrivals through the contact-gated
policy (gradients only generated while disconnected) and so ran the simulator at
a realized rate near 0.10 grad/slot instead of the model's lam.

Run:  python scripts/queueing_analysis.py

The default EnvConfig (eta=2.0) has all states net-negative in energy, so the
battery drains to 0 and P_ready -> 0. This script uses a calibrated config
(eta=4.0) with a proper stationary battery distribution. Calibration note:
choose eta so that eta*P(sunlit) - P_tx*P(C=1) - P_gpu*P(C=0) - P_base > 0.
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

from orbital_fl.analysis_config import ANALYSIS_CFG
from orbital_fl.markov import (
    run_stationary_analysis, solve_queue_chain, offperiod_moments,
)
from orbital_fl.queueing import (
    QueueParams, mean_staleness, mean_staleness_batch, mean_staleness_general,
    is_stable,
)
from orbital_fl.queue_sim import server_ready_path, simulate_queue_on_path


NOMINAL_LAM = 0.5
N_SEEDS = 16
SIM_HORIZON = 20000
SWEEP_LAM = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SOLVE_Q_MAX = 600          # truncation for the exact solve (overflow checked)


def empirical_staleness_ci(ready_paths, lam, mu, seeds):
    """Per-seed mean staleness on cached server paths; returns (mean, ci95)."""
    per_seed = []
    for sd, ready in zip(seeds, ready_paths):
        out = simulate_queue_on_path(ready, lam, mu, seed=1000 + sd)
        per_seed.append(out["mean_staleness"])
    arr = np.array(per_seed)
    sem = np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(1.96 * sem)


def main():
    cfg = ANALYSIS_CFG
    mu = int(cfg.mu)

    # --- 1. energy-channel stationary analysis --- #
    res = run_stationary_analysis(cfg)
    p_ready, beta = res["p_ready"], res["beta"]
    om = offperiod_moments(cfg)

    print(f"P_ready             : {p_ready:.4f}")
    print(f"beta (recovery)     : {beta:.4f}   (geometric off-period mean 1/beta = {om['geom_mean']:.2f})")
    print(f"off-period E[V]     : {om['E_V']:.2f}   E[V^2] = {om['E_V2']:.1f}   cv^2 = {om['cv2']:.2f}")
    print(f"off-period residual : {om['residual']:.2f}   (geometric residual = {(2-beta)/(2*beta):.2f})")

    # --- 2. cache server on/off paths once; reuse across all lam --- #
    seeds = list(range(N_SEEDS))
    print(f"\nSampling {N_SEEDS} server paths ({SIM_HORIZON} slots each)...", flush=True)
    ready_paths = [server_ready_path(cfg, SIM_HORIZON, seed=sd) for sd in seeds]

    # --- 3. nominal-load comparison --- #
    q = QueueParams(lam=NOMINAL_LAM, mu=mu, p_ready=p_ready, beta=beta)
    rho_eff = NOMINAL_LAM / (mu * p_ready)
    t_geom = mean_staleness(q)
    t_corr = mean_staleness_general(NOMINAL_LAM, mu, om["E_V"], om["E_V2"])
    sol = solve_queue_chain(cfg, NOMINAL_LAM, mu, Q_max=SOLVE_Q_MAX)
    t_exact = sol["E_T"]
    t_emp, ci_emp = empirical_staleness_ci(ready_paths, NOMINAL_LAM, mu, seeds)

    print(f"\n--- Mean staleness at lam={NOMINAL_LAM} (rho_eff={rho_eff:.3f}) ---")
    print(f"  vacation (geometric beta)        : {t_geom:.2f} slots")
    print(f"  vacation (true off-period moment): {t_corr:.2f} slots")
    print(f"  exact joint Markov-modulated     : {t_exact:.2f} slots  (overflow {sol['overflow_mass']:.1e})")
    print(f"  empirical ({N_SEEDS}-seed Poisson rollout) : {t_emp:.2f} +/- {ci_emp:.2f} slots")

    # --- 4. load sweep --- #
    print("\n--- Staleness vs load ---")
    print(f"{'lam':>5} {'rho_eff':>8} {'geom':>8} {'exact':>8} {'empirical (95% CI)':>22}")
    sweep = []
    for lam in SWEEP_LAM:
        qq = QueueParams(lam=lam, mu=mu, p_ready=p_ready, beta=beta)
        re = lam / (mu * p_ready)
        tg = mean_staleness(qq) if is_stable(qq) else float("nan")
        so = solve_queue_chain(cfg, lam, mu, Q_max=SOLVE_Q_MAX)
        te = so["E_T"]
        tm, cm = empirical_staleness_ci(ready_paths, lam, mu, seeds)
        sweep.append((lam, re, tg, te, tm, cm, so["overflow_mass"]))
        print(f"{lam:5.2f} {re:8.3f} {tg:8.2f} {te:8.2f}     {tm:7.2f} +/- {cm:5.2f}")

    sweep = np.array([(s[0], s[1], s[2], s[3], s[4], s[5]) for s in sweep])

    # --- 5. figure --- #
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Gradient queue analysis: Markov-modulated staleness", fontsize=13)

    # (a) battery stationary distribution
    ax = axes[0, 0]
    ax.bar(range(cfg.B_max + 1), res["marginal_B"], color="steelblue", alpha=0.85)
    ax.axvline(cfg.B_crit, color="red", linestyle="--", linewidth=1.2,
               label=f"$B_{{\\mathrm{{crit}}}} = {cfg.B_crit}$")
    ax.set_xlabel("Battery level $B$")
    ax.set_ylabel("Stationary probability")
    ax.set_title(f"(a) Battery distribution ($P_{{\\mathrm{{ready}}}}={p_ready:.3f}$)")
    ax.legend(fontsize=9)

    # (b) staleness vs load: three models + empirical
    ax = axes[0, 1]
    ax.plot(sweep[:, 1], sweep[:, 2], "s--", color="darkorange",
            label="vacation (geometric $\\beta$)", markersize=5)
    ax.plot(sweep[:, 1], sweep[:, 3], "o-", color="steelblue",
            label="exact Markov-modulated", markersize=5)
    ax.errorbar(sweep[:, 1], sweep[:, 4], yerr=sweep[:, 5], fmt="^", color="black",
                capsize=3, markersize=5, label="simulation (95% CI)")
    ax.set_xlabel("Effective load $\\rho_{\\mathrm{eff}} = \\lambda/(\\mu P_{\\mathrm{ready}})$")
    ax.set_ylabel("Mean staleness $E[T]$ (slots)")
    ax.set_title("(b) Staleness vs load: models vs simulation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) backlog distribution from the exact solve (heavy tail / overflow decay)
    ax = axes[1, 0]
    marg_q = sol["marginal_Q"]
    qx = np.arange(len(marg_q))
    tail = marg_q[::-1].cumsum()[::-1]      # P[Q >= x]
    ax.semilogy(qx, np.clip(tail, 1e-9, 1), color="seagreen")
    ax.set_xlim(0, min(200, len(marg_q)))
    ax.set_xlabel("Backlog $x$ (gradients)")
    ax.set_ylabel("$P[Q \\geq x]$")
    ax.set_title(f"(c) Backlog tail at $\\lambda={NOMINAL_LAM}$ (exact solve)")
    ax.grid(True, which="both", alpha=0.3)

    # (d) DiLoCo batch penalty on top of the exact baseline
    ax = axes[1, 1]
    k_vals = np.arange(1, 16)
    t_batch = [t_exact + (k - 1) / (2.0 * (mu - NOMINAL_LAM)) for k in k_vals]
    ax.bar(k_vals, t_batch, color="steelblue", alpha=0.85)
    ax.axhline(t_exact, color="red", linestyle="--", linewidth=1.2,
               label=f"$k=1$ exact ({t_exact:.1f} slots)")
    ax.set_xlabel("DiLoCo batch size $k$")
    ax.set_ylabel("Mean staleness (slots)")
    ax.set_title(f"(d) DiLoCo batching penalty ($\\rho_{{\\mathrm{{eff}}}}={rho_eff:.2f}$)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.abspath(os.path.join(HERE, "..", "figures", "queueing_analysis.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
