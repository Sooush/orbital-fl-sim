"""
Baseline policies for the SatelliteEnv.

Each policy is a callable: obs (dict) -> (a_local, a_tx).
These are the comparison points the CMDP solver has to beat.
"""

from __future__ import annotations
from typing import Callable, Tuple

Action = Tuple[float, float]
Policy = Callable[[dict], Action]


def power_oblivious(obs: dict) -> Action:
    """Always do maximum local training, always claim full bandwidth.

    This is the strawman baseline. It ignores battery state entirely
    and will violate energy neutrality in any meaningful regime.
    """
    return (1.0, 1.0)


def make_power_weighted(B_crit: int = 4, B_max: int = 20) -> Policy:
    """Scale local-step intensity by available battery headroom.

    Heuristic: a_local = max(0, (B - B_crit) / (B_max - B_crit))
    a_tx = 1.0 when connected and above B_crit, else 0.
    """
    def policy(obs: dict) -> Action:
        B = obs["B"]
        C = obs["C"]
        headroom = max(0.0, (B - B_crit) / max(1, B_max - B_crit))
        a_local = headroom
        a_tx = 1.0 if (C == 1 and B > B_crit) else 0.0
        return (a_local, a_tx)
    return policy


def contact_gated(obs: dict) -> Action:
    """Original paper formulation: train iff disconnected, transmit iff connected.

    This is the policy that recovers Nimalan's stationary-analysis equation
    exactly. Useful as a sanity check.
    """
    C = obs["C"]
    return (1.0 - C, float(C))
