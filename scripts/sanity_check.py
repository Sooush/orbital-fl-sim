"""
Sanity check with the gradient queue in place.

Rolls each baseline policy and plots:
  - Battery
  - Illumination (shared exogenous chain, plotted once)
  - Contact (shared exogenous chain, plotted once)
  - Queue length
  - Cumulative gradients drained (= cumulative real reward)

Console output reports: avg throughput, queue stats, drop rate, mean staleness.

Run:  python scripts/sanity_check.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from orbital_fl import (
    SatelliteEnv, EnvConfig,
    power_oblivious, make_power_weighted, contact_gated,
)


def rollout(env: SatelliteEnv, policy, horizon: int) -> dict:
    B = np.zeros(horizon, dtype=np.int32)
    I = np.zeros(horizon, dtype=np.int32)
    C = np.zeros(horizon, dtype=np.int32)
    Q = np.zeros(horizon, dtype=np.int32)
    R = np.zeros(horizon, dtype=np.float32)
    cost = np.zeros(horizon, dtype=np.float32)
    all_drain_events = []   # list of DrainEvent across the rollout

    obs = env.reset(seed=42)
    for t in range(horizon):
        action = policy(obs)
        next_obs, reward, _, info = env.step(action)
        B[t] = info["B"]
        I[t] = info["I"]
        C[t] = info["C"]
        Q[t] = info["Q"]
        R[t] = reward
        cost[t] = info["cost"]
        all_drain_events.extend(info["drained_events"])
        obs = next_obs

    staleness = np.array([e.staleness for e in all_drain_events], dtype=np.int32) \
                if all_drain_events else np.array([0])

    return {
        "B": B, "I": I, "C": C, "Q": Q, "R": R, "cost": cost,
        "drain_events": all_drain_events,
        "total_arrived": env.total_arrived,
        "total_drained": env.total_drained,
        "total_dropped": env.total_dropped,
        "mean_staleness": float(staleness.mean()),
        "max_staleness": int(staleness.max()),
    }


def main():
    cfg = EnvConfig(seed=0)
    horizon = 1000

    policies = {
        "power_oblivious":  power_oblivious,
        "contact_gated":    contact_gated,
        "power_weighted":   make_power_weighted(B_crit=cfg.B_crit, B_max=cfg.B_max),
    }

    results = {}
    print(f"{'policy':20s}  {'throughput':>10s}  {'arrived':>8s}  {'drained':>8s}  "
          f"{'dropped':>8s}  {'avg_Q':>6s}  {'mean_stale':>10s}  {'cost_frac':>10s}")
    print("-" * 100)
    for name, pol in policies.items():
        env = SatelliteEnv(config=cfg)
        r = rollout(env, pol, horizon)
        results[name] = r
        avg_throughput = r["R"].mean()
        avg_Q = r["Q"].mean()
        cost_frac = r["cost"].mean()
        print(
            f"{name:20s}  "
            f"{avg_throughput:>10.3f}  "
            f"{r['total_arrived']:>8d}  "
            f"{r['total_drained']:>8d}  "
            f"{r['total_dropped']:>8d}  "
            f"{avg_Q:>6.2f}  "
            f"{r['mean_staleness']:>10.2f}  "
            f"{cost_frac:>10.3f}"
        )

    # --- plot --- #
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    t = np.arange(horizon)
    ref = results["power_oblivious"]

    # Battery
    ax = axes[0]
    for name, r in results.items():
        ax.plot(t, r["B"], label=name, alpha=0.8)
    ax.axhline(cfg.B_crit, color="red", linestyle="--", linewidth=0.8,
               label=f"B_crit={cfg.B_crit}")
    ax.set_ylabel("Battery B_t")
    ax.set_title("Single-satellite env with gradient queue")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Illumination
    ax = axes[1]
    ax.fill_between(t, 0, 1, where=ref["I"] == 1, alpha=0.3, label="sunlit")
    ax.set_ylabel("I_t")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Contact
    ax = axes[2]
    ax.fill_between(t, 0, 1, where=ref["C"] == 1, alpha=0.3,
                    color="green", label="ISL connected")
    ax.set_ylabel("C_t")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Queue length
    ax = axes[3]
    for name, r in results.items():
        ax.plot(t, r["Q"], label=name, alpha=0.8)
    ax.axhline(cfg.Q_max, color="red", linestyle="--", linewidth=0.8,
               label=f"Q_max={cfg.Q_max}")
    ax.set_ylabel("Queue Q_t")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cumulative gradients drained (= real throughput)
    ax = axes[4]
    for name, r in results.items():
        ax.plot(t, np.cumsum(r["R"]), label=name, alpha=0.8)
    ax.set_ylabel("Gradients drained (cumulative)")
    ax.set_xlabel("Timestep t")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.abspath(os.path.join(HERE, "..", "figures", "sanity_check.png"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()