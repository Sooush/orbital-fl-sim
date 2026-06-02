"""
Stationary analysis of the per-satellite (B, I, C) Markov chain.

Builds the full transition matrix under the contact-gated policy,
solves for the stationary distribution, and extracts P_ready and
the effective link-recovery rate beta. Both quantities feed directly
into the queueing model in queueing.py.

This module also contains the exact joint (Q, B, I, C) Markov-modulated
queue solve. The buffer is served at rate mu whenever the satellite is in
the ready set and is off otherwise, so the gradient queue is a discrete-time
queue modulated by the energy-channel chain (the matrix-geometric / random-
environment setting from EE 384S Lec 3). Solving the joint chain gives the
exact mean backlog and, via Little's law, the exact mean staleness, which the
single-geometric vacation approximation in queueing.py only estimates.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .env import EnvConfig


def _idx(b: int, i: int, c: int) -> int:
    return b * 4 + i * 2 + c


def _poisson_pmf(lam: float, a_max: int) -> np.ndarray:
    """Truncated Poisson(lam) pmf on {0,...,a_max}; tail lumped into the last bin."""
    p = np.zeros(a_max + 1)
    p[0] = np.exp(-lam)
    for k in range(1, a_max + 1):
        p[k] = p[k - 1] * lam / k
    p[a_max] += max(0.0, 1.0 - p.sum())
    return p


def _env_transitions(cfg: EnvConfig, b: int, i: int, c: int):
    """
    Next (b', i', c') states and probabilities for the (B, I, C) chain under
    the contact-gated policy (train when disconnected, transmit when connected).
    Returns a list of (b', i', c', prob). Battery update is deterministic given
    (b, i, c); illumination and contact transition independently.
    """
    dE = cfg.eta * i - cfg.P_tx * c - cfg.P_gpu * (1 - c) - cfg.P_base
    b_next = int(np.clip(round(b + dE), 0, cfg.B_max))
    out = []
    for i2 in range(2):
        if i == 0:
            p_i = cfg.p_eclipse_exit if i2 == 1 else 1.0 - cfg.p_eclipse_exit
        else:
            p_i = cfg.p_eclipse_enter if i2 == 0 else 1.0 - cfg.p_eclipse_enter
        for c2 in range(2):
            if c == 0:
                p_c = cfg.p_link_acquire if c2 == 1 else 1.0 - cfg.p_link_acquire
            else:
                p_c = cfg.p_link_lose if c2 == 0 else 1.0 - cfg.p_link_lose
            out.append((b_next, i2, c2, p_i * p_c))
    return out


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


# ---------- exact joint (Q, B, I, C) Markov-modulated queue ----------------- #

def _qenc(q: int, b: int, i: int, c: int, B_max: int) -> int:
    return ((q * (B_max + 1) + b) * 2 + i) * 2 + c


def solve_queue_chain(cfg: EnvConfig, lam: float, mu: int,
                      Q_max: int = 250, a_max: int = 16) -> dict:
    """
    Exact stationary solve of the joint (Q, B, I, C) chain.

    Gradients arrive as an independent Poisson(lam) stream each slot. The server
    drains up to mu gradients in a slot when the satellite is ready
    (B > B_crit and C = 1) and zero otherwise, so within a slot the backlog
    evolves as Q' = clip(Q + A - mu*1[ready], 0, Q_max). The energy-channel
    sub-state (B, I, C) evolves under the contact-gated policy exactly as in
    run_stationary_analysis, and is independent of the backlog.

    Q_max here is a numerical truncation for the solve, not the env buffer; pick
    it large enough that the boundary mass is negligible (overflow_mass below).
    The chain is a quasi-birth-death process in the backlog level with the
    (B, I, C) energy-channel phase, so it is sparse and solved as a sparse
    stationary system rather than a dense one.

    Returns a dict with the stationary distribution, the mean backlog E[Q], the
    mean staleness E[T] = E[Q]/lam_eff (Little's law), the carried arrival rate
    lam_eff after truncation losses, the truncation overflow mass, and the
    marginal backlog pmf.
    """
    B_max = cfg.B_max
    pa = _poisson_pmf(lam, a_max)
    qn = Q_max + 1
    nph = (B_max + 1) * 4                       # phase count = |(B, I, C)|
    n = qn * nph

    # Phase transition Phi (the (B, I, C) chain), ordered by _idx(b, i, c).
    Phi = sp.csr_matrix(build_transition_matrix(cfg))

    # Ready-phase indicator on the diagonal (B > B_crit and C = 1).
    ready_diag = np.zeros(nph)
    for b in range(cfg.B_crit + 1, B_max + 1):
        for i in range(2):
            ready_diag[_idx(b, i, 1)] = 1.0
    D_R = sp.diags(ready_diag)
    D_N = sp.diags(1.0 - ready_diag)

    # Level transition blocks: within a slot, q -> clip(q + a - service, 0, Q_max),
    # with service = mu when the phase is ready and 0 otherwise.
    def level_matrix(service: int) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for q in range(qn):
            for a, pa_ in enumerate(pa):
                if pa_ < 1e-15:
                    continue
                q_next = min(max(q + a - service, 0), Q_max)
                rows.append(q); cols.append(q_next); vals.append(pa_)
        return sp.csr_matrix((vals, (rows, cols)), shape=(qn, qn))

    L_R = level_matrix(mu)
    L_N = level_matrix(0)

    # Full generator: ready phases use L_R, not-ready phases use L_N, then the
    # phase advances by Phi. State index = q * nph + phase, matching _qenc.
    P = sp.kron(L_R, D_R @ Phi, format="csr") + sp.kron(L_N, D_N @ Phi, format="csr")

    # Solve pi (P - I) = 0 with a normalization row.
    A = (P.T - sp.identity(n, format="csr")).tolil()
    A[-1, :] = 1.0
    A = A.tocsr()
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    pi = spla.spsolve(A, rhs)
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()

    marg_Q = np.array([
        sum(pi[_qenc(q, b, i, c, B_max)]
            for b in range(B_max + 1) for i in range(2) for c in range(2))
        for q in range(qn)
    ])
    E_Q = float(np.sum(np.arange(qn) * marg_Q))
    overflow_mass = float(marg_Q[Q_max])

    # Carried load: arrivals are dropped only when the truncation boundary is hit,
    # which is negligible for a well-chosen Q_max. lam_eff corrects Little's law.
    lam_eff = lam * (1.0 - overflow_mass)
    E_T = E_Q / lam_eff if lam_eff > 0 else float("inf")

    return {
        "pi": pi,
        "E_Q": E_Q,
        "E_T": E_T,
        "lam_eff": lam_eff,
        "overflow_mass": overflow_mass,
        "marginal_Q": marg_Q,
    }


def offperiod_moments(cfg: EnvConfig) -> dict:
    """
    First two moments of the server off-period (the sojourn in the not-ready set
    of the (B, I, C) chain), seen by a slot that enters the off-period.

    The not-ready states form a transient class with substochastic transition
    block T. Starting from the stationary entry distribution into the not-ready
    set, the off-period length V has
        E[V]   = e . (I - T)^{-1} . 1
        E[V^2] = e . (2 (I - T)^{-1} - I) (I - T)^{-1} . 1
    where e is the (normalized) entry distribution. The off-period is genuinely
    two-phase: a fast contact-limited branch (link re-acquired) and a slow
    battery-limited branch (recharge above B_crit), so V is far more dispersed
    than the single geometric of rate beta. The residual off-period seen by an
    arriving gradient is E[V^2] / (2 E[V]), the quantity that enters the
    vacation staleness term, and it is much larger than (2-beta)/(2 beta).

    Returns E[V], E[V^2], the residual, the implied geometric mean 1/beta for
    comparison, and the squared coefficient of variation of V.
    """
    P = build_transition_matrix(cfg)
    pi = stationary_distribution(P)
    B_max = cfg.B_max

    ready = {
        _idx(b, i, 1)
        for b in range(cfg.B_crit + 1, B_max + 1)
        for i in range(2)
    }
    not_ready = sorted(s for s in range(P.shape[0]) if s not in ready)
    pos = {s: k for k, s in enumerate(not_ready)}
    m = len(not_ready)

    # Substochastic block among not-ready states.
    T = np.array([[P[s, s2] for s2 in not_ready] for s in not_ready])

    # Entry distribution: stationary flow from ready into each not-ready state,
    # normalized. This is the distribution of the off-period's first slot.
    entry = np.zeros(m)
    for s_ready in ready:
        for s2 in not_ready:
            entry[pos[s2]] += pi[s_ready] * P[s_ready, s2]
    if entry.sum() <= 0:
        return {"E_V": float("inf"), "E_V2": float("inf"),
                "residual": float("inf"), "geom_mean": float("inf"), "cv2": float("inf")}
    entry /= entry.sum()

    I = np.eye(m)
    N = np.linalg.inv(I - T)             # fundamental matrix
    ones = np.ones(m)
    E_V = float(entry @ N @ ones)
    E_V2 = float(entry @ (2.0 * N - I) @ N @ ones)
    residual = E_V2 / (2.0 * E_V)
    var_V = E_V2 - E_V ** 2
    cv2 = var_V / (E_V ** 2)

    beta = compute_beta(pi, P, cfg)
    return {
        "E_V": E_V,
        "E_V2": E_V2,
        "residual": residual,
        "geom_mean": 1.0 / beta if beta > 0 else float("inf"),
        "cv2": cv2,
    }
