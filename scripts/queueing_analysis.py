"""
Queueing analysis for the Federated Orbital Edge Mesh project.

Runs the full pipeline:
  1. Build and solve the (B, I, C) Markov chain for a calibrated config.
  2. Extract P_ready and beta.
  3. Compute analytical mean staleness (vacation decomposition).
  4. Validate against empirical DrainEvents from env.py rollouts.
  5. Generate figures/queueing_analysis.png with four subplots.

Run:  python scripts/queueing_analysis.py

The default EnvConfig (eta=2.0) has all states net-negative in energy,
so the battery drains to 0 and P_ready -> 0 in the long run. This script
uses a calibrated config (eta=4.0) where the battery has a proper
stationary distribution. Calibration note: choose eta so that
E[dE] = eta * P(sunlit) - P_tx*P(C=1) - P_gpu*P(C=0) - P_base > 0.
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

from orbital_fl import SatelliteEnv, EnvConfig, contact_gated
from orbital_fl.markov import run_stationary_analysis
from orbital_fl.queueing import (
    QueueParams, mean_staleness, mean_staleness_batch,
    staleness_vs_pready, staleness_vs_load,
    effective_rate_approx, is_stable,
)


# ---------- calibrated config ---------------------------------------------- #
# eta=4.0 gives E[dE] > 0 (battery charges when sunlit), producing a
# non-degenerate stationary distribution and a meaningful P_ready.
ANALYSIS_CFG = EnvConfig(
    B_max=20, B_crit=4, B_init=10,
    eta=4.0,
    P_base=0.5, P_gpu=2.0, P_tx=1.5,
    p_eclipse_exit=0.028, p_eclipse_enter=0.018,
    p_link_acquire=0.10, p_link_lose=0.05,
    lambda_g=0.5, mu=2, Q_max=50,
    seed=42,
)


def empirical_staleness(cfg: EnvConfig, horizon: int = 8000) -> float:
    """Roll out the contact_gated policy and return mean DrainEvent staleness."""
    env = SatelliteEnv(config=cfg)
    obs = env.reset(seed=0)
    staleness_list = []
    for _ in range(horizon):
        action = contact_gated(obs)
        obs, _, _, info = env.step(action)
        for ev in info["drained_events"]:
            staleness_list.append(ev.staleness)
    if not staleness_list:
        return float("nan")
    return float(np.mean(staleness_list))


def main():
    cfg = ANALYSIS_CFG

    # --- 1. Markov chain stationary analysis --- #
    result = run_stationary_analysis(cfg)
    p_ready = result["p_ready"]
    beta    = result["beta"]
    pi      = result["pi"]

    lam = cfg.lambda_g  # arrival rate at a_local = 1 (contact_gated: train when C=0)
    mu  = float(cfg.mu)

    q_nominal = QueueParams(lam=lam, mu=mu, p_ready=p_ready, beta=beta)

    print(f"P_ready          : {p_ready:.4f}")
    print(f"beta (recovery)  : {beta:.4f}")
    print(f"rho_eff          : {lam / (mu * p_ready):.4f}  ({'stable' if is_stable(q_nominal) else 'UNSTABLE'})")
    print(f"E[T] analytical  : {mean_staleness(q_nominal):.2f} slots")
    print(f"E[T] eff-rate approx: {effective_rate_approx(q_nominal):.2f} slots")

    emp = empirical_staleness(cfg)
    print(f"E[T] empirical   : {emp:.2f} slots  (8000-step rollout)")

    # --- 2. Plots --- #
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Gradient Queue Analysis — Federated Orbital Edge Mesh", fontsize=13)

    # (a) Battery stationary distribution
    ax = axes[0, 0]
    ax.bar(range(cfg.B_max + 1), result["marginal_B"], color="steelblue", alpha=0.8)
    ax.axvline(cfg.B_crit, color="red", linestyle="--", linewidth=1.2,
               label=f"B_crit = {cfg.B_crit}")
    ax.set_xlabel("Battery level $B$")
    ax.set_ylabel("Stationary probability $\\pi_B$")
    ax.set_title(f"(a) Battery distribution  ($P_{{\\mathrm{{ready}}}} = {p_ready:.3f}$)")
    ax.legend(fontsize=9)

    # (b) Mean staleness vs P_ready (at fixed rho_eff)
    ax = axes[0, 1]
    p_arr, _ = staleness_vs_pready(lam, mu, beta)
    for rho_target in [0.3, 0.5, 0.7]:
        lam_t = rho_target * mu * p_arr
        vals = []
        for p, l in zip(p_arr, lam_t):
            q = QueueParams(lam=l, mu=mu, p_ready=p, beta=beta)
            vals.append(mean_staleness(q) if is_stable(q) else float("nan"))
        ax.semilogy(p_arr, vals, label=f"$\\rho_{{\\mathrm{{eff}}}} = {rho_target}$")
    ax.axvline(p_ready, color="gray", linestyle=":", linewidth=1,
               label=f"Config $P_{{\\mathrm{{ready}}}}$")
    ax.set_xlabel("$P_{\\mathrm{ready}}$")
    ax.set_ylabel("Mean staleness $\\mathbb{E}[T]$ (slots)")
    ax.set_title("(b) Staleness vs. server uptime")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # (c) Mean staleness vs effective load (at fixed P_ready)
    ax = axes[1, 0]
    for p in [0.4, p_ready, 0.8]:
        rho_range, t_vals = staleness_vs_load(mu, p, beta)
        ax.semilogy(rho_range, t_vals, label=f"$P_{{\\mathrm{{ready}}}} = {p:.2f}$")
    # Also plot effective-rate approximation for config P_ready
    rho_range_base = np.linspace(0.05, 0.95, 300)
    t_eff = []
    for rho in rho_range_base:
        lam_t = rho * mu * p_ready
        q = QueueParams(lam=lam_t, mu=mu, p_ready=p_ready, beta=beta)
        t_eff.append(effective_rate_approx(q) if is_stable(q) else float("nan"))
    ax.semilogy(rho_range_base, t_eff, "k--", linewidth=1,
                label="Eff.-rate approx.")
    if is_stable(q_nominal):
        rho_nom = lam / (mu * p_ready)
        ax.axvline(rho_nom, color="gray", linestyle=":", linewidth=1,
                   label=f"Config $\\rho_{{\\mathrm{{eff}}}}$")
    ax.set_xlabel("Effective load $\\rho_{\\mathrm{eff}}$")
    ax.set_ylabel("Mean staleness $\\mathbb{E}[T]$ (slots)")
    ax.set_title("(c) Staleness vs. load (vacation vs. eff.-rate approx.)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (d) DiLoCo batch penalty vs batch size k
    ax = axes[1, 1]
    if is_stable(q_nominal):
        k_vals = np.arange(1, 16)
        t_batch = [mean_staleness_batch(q_nominal, int(k)) for k in k_vals]
        t_base  = mean_staleness(q_nominal)
        ax.bar(k_vals, t_batch, color="steelblue", alpha=0.8, label="$\\mathbb{E}[T]_{\\mathrm{batch}}$")
        ax.axhline(t_base, color="red", linestyle="--", linewidth=1.2,
                   label=f"$k=1$ baseline ({t_base:.1f} slots)")
        ax.set_xlabel("DiLoCo batch size $k$")
        ax.set_ylabel("Mean staleness (slots)")
        ax.set_title("(d) DiLoCo batching penalty ($\\rho_{\\mathrm{eff}}$"
                     f" $= {lam/(mu*p_ready):.2f}$)")
        ax.legend(fontsize=9)
        # Annotate empirical point
        ax.annotate(f"Empirical\n(k=1): {emp:.1f}",
                    xy=(1, emp), xytext=(3, emp * 1.1),
                    arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    else:
        ax.text(0.5, 0.5, "Unstable regime", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    out = os.path.join(HERE, "..", "figures", "queueing_analysis.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"\nSaved figure: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
