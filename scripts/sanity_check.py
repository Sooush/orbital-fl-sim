"""
Sanity check: roll out each baseline policy and plot battery / illumination /
contact / cumulative reward.

This is the "does the env even work" plot. If battery stays bounded, eclipse
cycles look periodic-ish, and the policies behave differently, we have a
working substrate.

Run:  python scripts/sanity_check.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Make src importable when running this script directly.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from orbital_fl import (
    SatelliteEnv, EnvConfig,
    power_oblivious, make_power_weighted, contact_gated,
)


def rollout(env: SatelliteEnv, policy, horizon: int) -> dict:
    """Run policy for `horizon` steps, return arrays of state and reward."""
    B = np.zeros(horizon, dtype=np.int32)
    I = np.zeros(horizon, dtype=np.int32)
    C = np.zeros(horizon, dtype=np.int32)
    R = np.zeros(horizon, dtype=np.float32)
    a_local_arr = np.zeros(horizon, dtype=np.float32)
    a_tx_arr = np.zeros(horizon, dtype=np.float32)

    obs = env.reset(seed=42)
    for t in range(horizon):
        action = policy(obs)
        next_obs, reward, _, info = env.step(action)
        B[t] = info["B"]
        I[t] = info["I"]
        C[t] = info["C"]
        R[t] = reward
        a_local_arr[t] = info["a_local"]
        a_tx_arr[t] = info["a_tx"]
        obs = next_obs

    return {
        "B": B, "I": I, "C": C, "R": R,
        "a_local": a_local_arr, "a_tx": a_tx_arr,
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
    for name, pol in policies.items():
        env = SatelliteEnv(config=cfg)
        results[name] = rollout(env, pol, horizon)
        avg_reward = results[name]["R"].mean()
        battery_dead_frac = (results[name]["B"] == 0).mean()
        print(
            f"{name:20s}  avg_reward={avg_reward:.3f}  "
            f"battery_dead_frac={battery_dead_frac:.3f}  "
            f"avg_battery={results[name]['B'].mean():.2f}"
        )

    # --- plot --- #
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    t = np.arange(horizon)

    # Battery
    ax = axes[0]
    for name, r in results.items():
        ax.plot(t, r["B"], label=name, alpha=0.8)
    ax.axhline(cfg.B_crit, color="red", linestyle="--", linewidth=0.8,
               label=f"B_crit={cfg.B_crit}")
    ax.set_ylabel("Battery B_t")
    ax.set_title("Single-satellite environment sanity check")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Illumination (same exogenous chain for all policies, plot once)
    ax = axes[1]
    ref = results["power_oblivious"]
    ax.fill_between(t, 0, 1, where=ref["I"] == 1, alpha=0.3, label="sunlit")
    ax.set_ylabel("Illumination I_t")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Contact
    ax = axes[2]
    ax.fill_between(t, 0, 1, where=ref["C"] == 1, alpha=0.3,
                    color="green", label="ISL connected")
    ax.set_ylabel("Contact C_t")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cumulative reward
    ax = axes[3]
    for name, r in results.items():
        ax.plot(t, np.cumsum(r["R"]), label=name, alpha=0.8)
    ax.set_ylabel("Cumulative reward")
    ax.set_xlabel("Timestep t")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(HERE, "..", "figures", "sanity_check.png")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()
