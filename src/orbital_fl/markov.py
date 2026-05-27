"""
Stationary analysis of the per-satellite (B, I, C) Markov chain.

Builds the full transition matrix under the contact-gated policy,
solves for the stationary distribution, and extracts P_ready and
the effective link-recovery rate beta. Both quantities feed directly
into the queueing model in queueing.py.
"""

from __future__ import annotations

import numpy as np

from .env import EnvConfig


def _idx(b: int, i: int, c: int) -> int:
    return b * 4 + i * 2 + c


def build_transition_matrix(cfg: EnvConfig) -> np.ndarray:
    """
    Build the (B_max+1)*4 x (B_max+1)*4 transition matrix P for the
    joint (B, I, C) chain under the contact-gated policy
    (a_local = 1-c, a_tx = c).

    State ordering: index(b, i, c) = b*4 + i*2 + c
    Rows index current state; columns index next state.
    """
    B_max = cfg.B_max
    n = (B_max + 1) * 4
    P = np.zeros((n, n))

    for b in range(B_max + 1):
        for i in range(2):
            for c in range(2):
                # Energy update (contact-gated: train when C=0, transmit when C=1)
                dE = (
                    cfg.eta * i
                    - cfg.P_tx * c
                    - cfg.P_gpu * (1 - c)
                    - cfg.P_base
                )
                b_next = int(np.clip(round(b + dE), 0, B_max))
                s = _idx(b, i, c)

                for i_next in range(2):
                    # P(I_next | I)
                    if i == 0:  # eclipsed
                        p_i = cfg.p_eclipse_exit if i_next == 1 else (1.0 - cfg.p_eclipse_exit)
                    else:       # sunlit
                        p_i = cfg.p_eclipse_enter if i_next == 0 else (1.0 - cfg.p_eclipse_enter)

                    for c_next in range(2):
                        # P(C_next | C)
                        if c == 0:  # disconnected
                            p_c = cfg.p_link_acquire if c_next == 1 else (1.0 - cfg.p_link_acquire)
                        else:       # connected
                            p_c = cfg.p_link_lose if c_next == 0 else (1.0 - cfg.p_link_lose)

                        P[s, _idx(b_next, i_next, c_next)] += p_i * p_c

    return P


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Solve pi @ P = pi, sum(pi) = 1 by replacing the last balance equation
    with the normalization constraint.
    """
    n = P.shape[0]
    A = (P.T - np.eye(n)).astype(float)
    b = np.zeros(n)
    A[-1, :] = 1.0
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return pi


def compute_pready(pi: np.ndarray, cfg: EnvConfig) -> float:
    """
    P_ready = P(B > B_crit, C = 1)

    The marginal probability that the satellite has sufficient charge
    and an active ISL to push gradient updates. Uses strict inequality
    B > B_crit to match the `had_energy = B > B_crit` gate in env.py.
    """
    p = 0.0
    for b in range(cfg.B_crit + 1, cfg.B_max + 1):  # strictly greater than B_crit
        for i in range(2):
            p += pi[_idx(b, i, 1)]  # c = 1
    return float(p)


def compute_beta(pi: np.ndarray, P: np.ndarray, cfg: EnvConfig) -> float:
    """
    Effective per-slot probability of transitioning from 'not ready' into
    the ready set.

    beta = sum_{s not ready, s' ready} pi_s * P_{s,s'} / sum_{s not ready} pi_s

    This is the empirical recovery rate that parameterises the off-period
    distribution in the vacation queueing model.
    """
    B_max = cfg.B_max
    ready = {
        _idx(b, i, 1)
        for b in range(cfg.B_crit + 1, B_max + 1)  # strictly greater than B_crit
        for i in range(2)
    }
    not_ready = [s for s in range(P.shape[0]) if s not in ready]
    ready_list = list(ready)

    flow = sum(pi[s] * P[s, s2] for s in not_ready for s2 in ready_list)
    mass = sum(pi[s] for s in not_ready)
    return float(flow / max(mass, 1e-12))


def run_stationary_analysis(cfg: EnvConfig) -> dict:
    """
    Full stationary analysis for a given EnvConfig. Returns a dict with:
        P         -- transition matrix
        pi        -- stationary distribution vector
        p_ready   -- scalar P_ready
        beta      -- effective recovery rate
        marginal_B -- marginal battery PMF (length B_max+1)
        marginal_I -- marginal illumination PMF (length 2)
        marginal_C -- marginal contact PMF (length 2)
    """
    P_mat = build_transition_matrix(cfg)
    pi = stationary_distribution(P_mat)
    p_ready = compute_pready(pi, cfg)
    beta = compute_beta(pi, P_mat, cfg)

    B_max = cfg.B_max
    marginal_B = np.array([
        sum(pi[_idx(b, i, c)] for i in range(2) for c in range(2))
        for b in range(B_max + 1)
    ])
    marginal_I = np.array([
        sum(pi[_idx(b, i, c)] for b in range(B_max + 1) for c in range(2))
        for i in range(2)
    ])
    marginal_C = np.array([
        sum(pi[_idx(b, i, c)] for b in range(B_max + 1) for i in range(2))
        for c in range(2)
    ])

    return {
        "P": P_mat,
        "pi": pi,
        "p_ready": p_ready,
        "beta": beta,
        "marginal_B": marginal_B,
        "marginal_I": marginal_I,
        "marginal_C": marginal_C,
    }
