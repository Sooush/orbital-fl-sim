"""
Robustness sweeps: gradient compression and ISL fading limits.

The body of the paper promises staleness as a function of gradient compression
ratio and identification of the fading limit where the queue breaks down. This
script delivers both, using the exact Markov-modulated solve.

  Compression. A compression ratio r lets the link push r times as many gradients
  per slot, i.e. an effective service rate mu_eff = r. Sweeping mu_eff shows that
  staleness saturates at a floor set by the off-period wait, which compression
  cannot remove because the link is still unavailable for the same fraction of time.

  Fading. Increasing the ISL link-loss probability p_link_lose lowers P_ready and
  raises the effective load. The fading limit is the link-loss rate at which
  rho_eff reaches 1 and the gradient queue goes unstable.

Run:  python scripts/robustness_sweeps.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from orbital_fl.markov import run_stationary_analysis, solve_queue_chain
from orbital_fl.analysis_config import ANALYSIS_CFG

LAM = 0.5
MU_VALUES = [2, 3, 4, 5, 6, 8, 10]              # compression ratio proxy (mu=1 is unstable)
PLOSS_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]


def main():
    cfg = ANALYSIS_CFG

    # --- compression sweep --- #
    # The staleness floor is the mu -> infinity limit: even with instant service,
    # a gradient still waits for the link to become ready. We compute it directly
    # by solving the chain with a very large drain rate (whole queue cleared each
    # ready slot), which isolates the irreducible off-period wait.
    print("Compression sweep (mu_eff = effective drain capacity):")
    floor = solve_queue_chain(cfg, LAM, 50, Q_max=600)["E_T"]
    comp_T = []
    for mu in MU_VALUES:
        sol = solve_queue_chain(cfg, LAM, mu, Q_max=600)
        comp_T.append(sol["E_T"])
        print(f"  mu_eff={mu}  E[T]={sol['E_T']:6.2f}  (instant-service floor ~ {floor:.1f})")

    # --- fading sweep --- #
    print("\nFading sweep (ISL link-loss probability):")
    fad_pready, fad_rho, fad_T = [], [], []
    for pl in PLOSS_VALUES:
        cfg_f = replace(cfg, p_link_lose=pl)
        res = run_stationary_analysis(cfg_f)
        pr = res["p_ready"]
        rho = LAM / (cfg.mu * pr)
        fad_pready.append(pr); fad_rho.append(rho)
        if rho < 1.0:
            sol = solve_queue_chain(cfg_f, LAM, int(cfg.mu), Q_max=800)
            T = sol["E_T"] if sol["overflow_mass"] < 1e-3 else float("nan")
        else:
            T = float("nan")
        fad_T.append(T)
        print(f"  p_link_lose={pl:.2f}  P_ready={pr:.3f}  rho_eff={rho:.3f}  "
              f"E[T]={T if not np.isnan(T) else float('inf'):.2f}")

    # fading limit = interpolated p_loss where rho_eff crosses 1 (P_ready = lam/mu)
    rho_arr = np.array(fad_rho)
    fading_limit = None
    for j in range(1, len(PLOSS_VALUES)):
        if rho_arr[j - 1] < 1.0 <= rho_arr[j]:
            frac = (1.0 - rho_arr[j - 1]) / (rho_arr[j] - rho_arr[j - 1])
            fading_limit = PLOSS_VALUES[j - 1] + frac * (PLOSS_VALUES[j] - PLOSS_VALUES[j - 1])
            break
    if fading_limit is not None:
        print(f"  fading limit (rho_eff = 1): p_loss ~ {fading_limit:.3f}")

    # --- figure --- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Robustness: gradient compression and ISL fading limits", fontsize=13)

    ax = axes[0]
    ax.plot(MU_VALUES, comp_T, "o-", color="steelblue", label="exact $E[T]$")
    ax.axhline(floor, color="red", linestyle="--", linewidth=1.2,
               label=f"instant-service floor ({floor:.1f} slots)")
    ax.set_xlabel("Compression ratio / effective service $\\mu_{\\mathrm{eff}}$")
    ax.set_ylabel("Mean staleness $E[T]$ (slots)")
    ax.set_title("(a) Compression has diminishing returns")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(PLOSS_VALUES, fad_pready, "o-", color="seagreen", label="$P_{\\mathrm{ready}}$")
    ax.axhline(LAM / cfg.mu, color="red", linestyle="--", linewidth=1.2,
               label=f"stability floor $\\lambda/\\mu = {LAM/cfg.mu:.2f}$")
    ax.set_xlabel("ISL link-loss probability $p_{\\mathrm{loss}}$")
    ax.set_ylabel("$P_{\\mathrm{ready}}$")
    ax.set_title("(b) Availability collapses with fading")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(PLOSS_VALUES, fad_T, "o-", color="firebrick", label="exact $E[T]$")
    if fading_limit is not None:
        ax.axvline(fading_limit, color="gray", linestyle="--", linewidth=1.2,
                   label=f"fading limit $\\approx {fading_limit:.2f}$")
    ax.set_xlabel("ISL link-loss probability $p_{\\mathrm{loss}}$")
    ax.set_ylabel("Mean staleness $E[T]$ (slots)")
    ax.set_title("(c) Staleness diverges at the fading limit")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.abspath(os.path.join(HERE, "..", "figures", "robustness_sweeps.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
