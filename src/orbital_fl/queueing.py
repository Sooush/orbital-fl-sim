"""
Analytical queueing model for gradient aggregation over intermittent ISLs.

Each satellite maintains a gradient buffer modelled as an M/M/1 queue with
a two-state on-off server. The server is 'on' (service rate mu) when the
satellite is in the ready state (C=1, B >= B_crit) and 'off' otherwise.
The steady-state server uptime is P_ready, computed by markov.py.

Key results implemented here:
  - Stability condition: rho_eff = lam / (mu * p_ready) < 1
  - Mean staleness via the M/G/1 vacation decomposition (Doshi 1986)
  - N-satellite Jackson product-form and its breakdown regimes
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class QueueParams:
    """Parameters for the single-satellite gradient queue."""
    lam: float        # gradient arrival rate (per slot)
    mu: float         # service rate when server is on (gradients per slot)
    p_ready: float    # steady-state server uptime (P_ready from Markov chain)
    beta: float       # per-slot probability of recovering the ready state
    n_sat: int = 1    # number of satellites (for constellation analysis)


# ---------- core analytical results --------------------------------------- #

def effective_load(q: QueueParams) -> float:
    """rho_eff = lam / (mu * p_ready). Stability requires rho_eff < 1."""
    return q.lam / (q.mu * q.p_ready)


def is_stable(q: QueueParams) -> bool:
    return effective_load(q) < 1.0 and q.lam < q.mu


def mean_staleness(q: QueueParams) -> float:
    """
    Mean gradient sojourn time E[T] (slots), equal to expected staleness
    at drain time (T_drain - T_arrival).

    Uses the M/G/1 multiple-vacation decomposition (Doshi 1986):

        E[T] = 1 / (mu - lam)  +  (2 - beta) / (2 * beta)

    The first term is the M/M/1 sojourn without intermittency (stable since
    lam < mu * p_ready <= mu). The second term is the residual off-period:
    the expected wait for the ISL to recover, averaged over arrival epochs.

    Off-periods are geometric with success probability beta, giving
    E[V] = 1/beta and E[V^2] = (2-beta)/beta^2, so E[V_res] = (2-beta)/(2*beta).
    """
    if not is_stable(q):
        return float("inf")
    t_mm1 = 1.0 / (q.mu - q.lam)
    t_vac = (2.0 - q.beta) / (2.0 * q.beta)
    return t_mm1 + t_vac


def mean_queue_length(q: QueueParams) -> float:
    """E[L] = lam * E[T]  (Little's law)."""
    return q.lam * mean_staleness(q)


def vacation_penalty(q: QueueParams) -> float:
    """Additive staleness cost due to ISL intermittency alone."""
    return (2.0 - q.beta) / (2.0 * q.beta)


def mean_staleness_general(lam: float, mu: float, e_v: float, e_v2: float) -> float:
    """
    Vacation staleness with an arbitrary off-period distribution.

    The single-geometric model in mean_staleness assumes the off-period is
    geometric with rate beta, giving residual (2-beta)/(2*beta). The true
    off-period of the energy-channel chain is two-phase (a fast contact-limited
    branch and a slow battery-limited branch), so its second moment e_v2 is far
    larger than a geometric of the same mean. Passing the exact moments
    (from markov.offperiod_moments) gives the corrected residual e_v2/(2*e_v):

        E[T] = 1/(mu - lam)  +  e_v2 / (2 * e_v)

    This narrows but does not close the gap to the exact joint-chain solve,
    because the M/G/1 decomposition still treats the in-service and off-period
    delays as separable, which the genuine queue-backlog correlation violates.
    """
    if lam >= mu:
        return float("inf")
    return 1.0 / (mu - lam) + e_v2 / (2.0 * e_v)


def discrete_time_backlog(lam: float, mu: float) -> float:
    """
    Mean backlog of the always-on discrete-time slotted queue (EE 384S Lec 5),

        B_bar = lam * (1 - mu) / (mu - lam),     lam < mu,

    where lam and mu are per-slot arrival and service probabilities. This is the
    no-intermittency reference: it is what the gradient buffer would experience
    if the ISL were always available (P_ready = 1). The gap between this and the
    intermittent-server staleness isolates the cost of ISL outages.
    """
    if lam >= mu:
        return float("inf")
    return lam * (1.0 - mu) / (mu - lam)


def mean_staleness_batch(q: QueueParams, k: int) -> float:
    """
    Mean staleness under DiLoCo-style batching: k local SGD steps per sync.
    Gradients arrive in batches of size k (M^[X]/M/1 queue with vacation).

    For M^[X]/M/1 (batch Poisson, rate Lambda=lam/k, batch size k, service
    rate mu), E[L] = rho/(1-rho) + Lambda*k*(k-1)/(2*mu*(1-rho)) where
    rho = lam/mu. Applying Little's law E[T] = E[L]/lam yields:

        E[T]_{batch} = (k+1) / (2*(mu-lam))  + vacation penalty

    so the correction over k=1 is:

        delta = (k-1) / (2*(mu - lam))

    Note: uses actual load rho = lam/mu, not rho_eff, because the batch
    correction is a property of the server capacity, not the server uptime.
    """
    if k <= 1:
        return mean_staleness(q)
    if not is_stable(q):
        return float("inf")
    pk_correction = (k - 1) / (2.0 * (q.mu - q.lam))
    return mean_staleness(q) + pk_correction


# ---------- sweep functions for plotting ---------------------------------- #

def staleness_vs_pready(
    lam: float,
    mu: float,
    beta: float,
    p_range: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical E[T] as P_ready is swept from 0 to 1."""
    if p_range is None:
        p_range = np.linspace(0.05, 0.99, 300)
    out = []
    for p in p_range:
        q = QueueParams(lam=lam, mu=mu, p_ready=p, beta=beta)
        out.append(mean_staleness(q) if is_stable(q) else float("nan"))
    return p_range, np.array(out)


def staleness_vs_load(
    mu: float,
    p_ready: float,
    beta: float,
    rho_range: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical E[T] as effective load rho_eff is swept from 0 to 1."""
    if rho_range is None:
        rho_range = np.linspace(0.05, 0.95, 300)
    lam_values = rho_range * mu * p_ready
    out = []
    for lam in lam_values:
        q = QueueParams(lam=lam, mu=mu, p_ready=p_ready, beta=beta)
        out.append(mean_staleness(q))
    return rho_range, np.array(out)


def effective_rate_approx(q: QueueParams) -> float:
    """
    Simpler approximation: treat server as always-on at effective rate
    mu * p_ready. Underestimates delay by ignoring vacation penalty.
    """
    if not is_stable(q):
        return float("inf")
    mu_eff = q.mu * q.p_ready
    return 1.0 / (mu_eff - q.lam)
