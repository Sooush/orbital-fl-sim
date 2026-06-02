"""
N-satellite constellation analysis: superposition of on/off gradient sources.

Each satellite is an energy-channel chain (B, I, C) under the contact-gated
policy with an independent Poisson(lam) gradient stream and a local buffer that
drains at rate mu while the satellite is ready. The aggregator collects the
drained gradients from all N satellites into one buffer. This is the discrete-
time analogue of the EE 384S Lec 3 example "superposition of N independent
on/off sources", and it lets us test two structural claims from the paper:

  1. Product form. With independent per-satellite chains the joint backlog
     distribution should factor, pi(q_1,...,q_N) = prod_i pi_i(q_i). We measure
     the total-variation distance between the empirical joint law of (Q_1, Q_2)
     and the product of its marginals.

  2. Merged arrival rate. In steady state the aggregator input rate equals the
     summed per-satellite throughput, which equals N*lam by flow balance.

The breakdown experiment shares the illumination (eclipse) chain across the
plane with tunable correlation rho_corr in [0,1]. ISL contact stays independent
per link. As rho_corr grows the per-satellite queues become positively
correlated and the product form fails, which is the regime the paper flags.

This module mirrors the (B, I, C) dynamics of env.py rather than importing the
single-satellite stepping, because the correlated-eclipse driver needs to inject
a shared illumination transition across satellites. The rho_corr=0 path is
checked against the env-based statistics in the analysis script.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .env import EnvConfig


@dataclass
class ConstellationResult:
    N: int
    lam: float
    rho_corr: float
    per_sat_throughput: float        # mean gradients drained per satellite per slot
    merged_rate: float               # aggregator input rate (grad/slot)
    queue_corr: float                # Pearson corr of (Q_1, Q_2) across time
    tv_distance: float               # TV( joint(Q_1,Q_2), marginal x marginal )
    agg_mean_delay: float            # mean aggregator sojourn (slots)
    agg_indep_delay: float           # M/M/1 prediction 1/(mu_agg - N*lam)
    mean_queue: float                # mean per-satellite backlog


def _illum_next(i: int, u: float, cfg: EnvConfig) -> int:
    """Two-state illumination transition driven by uniform u (mirrors env.py)."""
    if i == 0:
        return 1 if u < cfg.p_eclipse_exit else 0
    return 0 if u < cfg.p_eclipse_enter else 1


def _contact_next(c: int, u: float, cfg: EnvConfig) -> int:
    """Two-state ISL contact transition driven by uniform u (mirrors env.py)."""
    if c == 0:
        return 1 if u < cfg.p_link_acquire else 0
    return 0 if u < cfg.p_link_lose else 1


def simulate_constellation(cfg: EnvConfig, N: int, lam: float,
                           horizon: int = 20000, seed: int = 0,
                           rho_corr: float = 0.0, mu_agg: int | None = None,
                           warmup: int = 1000) -> ConstellationResult:
    """
    Roll N satellites and one aggregator.

    rho_corr in [0,1] is the per-slot probability that the whole plane shares a
    single illumination transition draw (synchronized eclipses); otherwise each
    satellite transitions illumination independently. mu_agg is the aggregator
    service rate (defaults to mu, matching one satellite's drain capacity).
    """
    rng = np.random.default_rng(seed)
    mu = int(cfg.mu)
    if mu_agg is None:
        mu_agg = mu

    B = np.full(N, cfg.B_init, dtype=int)
    I = np.full(N, cfg.I_init, dtype=int)
    C = np.full(N, cfg.C_init, dtype=int)
    Q = np.zeros(N, dtype=int)                  # per-satellite backlog
    agg = []                                    # aggregator FIFO of arrival slots

    q1_samples, q2_samples = [], []
    drained_total = 0
    agg_staleness = []
    qsum = 0
    nrec = 0

    for t in range(horizon):
        ready = (C == 1) & (B > cfg.B_crit)

        # arrivals + local service
        arr = rng.poisson(lam, size=N)
        Q = Q + arr
        served = np.where(ready, np.minimum(Q, mu), 0)
        Q = Q - served
        n_drained = int(served.sum())
        drained_total += n_drained

        # aggregator: drained gradients arrive now, drain up to mu_agg
        agg.extend([t] * n_drained)
        k = min(mu_agg, len(agg))
        for _ in range(k):
            a0 = agg.pop(0)
            if t >= warmup:
                agg_staleness.append(t - a0)

        if t >= warmup:
            q1_samples.append(int(Q[0]))
            if N > 1:
                q2_samples.append(int(Q[1]))
            qsum += int(Q.sum())
            nrec += 1

        # energy update (contact-gated) and exogenous transitions
        dE = cfg.eta * I - cfg.P_tx * C - cfg.P_gpu * (1 - C) - cfg.P_base
        B = np.clip(np.round(B + dE).astype(int), 0, cfg.B_max)

        share_illum = rng.random() < rho_corr
        if share_illum:
            u_shared = rng.random()
            I = np.array([_illum_next(int(I[j]), u_shared, cfg) for j in range(N)])
        else:
            us = rng.random(N)
            I = np.array([_illum_next(int(I[j]), us[j], cfg) for j in range(N)])
        uc = rng.random(N)
        C = np.array([_contact_next(int(C[j]), uc[j], cfg) for j in range(N)])

    per_sat_thr = drained_total / (N * horizon)
    merged_rate = drained_total / horizon
    mean_queue = qsum / max(nrec, 1) / N

    # product-form diagnostics on (Q_1, Q_2)
    if N > 1 and len(q2_samples) > 10:
        q1 = np.array(q1_samples); q2 = np.array(q2_samples)
        if q1.std() > 0 and q2.std() > 0:
            queue_corr = float(np.corrcoef(q1, q2)[0, 1])
        else:
            queue_corr = 0.0
        tv = _tv_distance_2d(q1, q2)
    else:
        queue_corr = float("nan")
        tv = float("nan")

    agg_delay = float(np.mean(agg_staleness)) if agg_staleness else float("nan")
    agg_indep = 1.0 / (mu_agg - N * lam) if mu_agg > N * lam else float("inf")

    return ConstellationResult(
        N=N, lam=lam, rho_corr=rho_corr,
        per_sat_throughput=per_sat_thr,
        merged_rate=merged_rate,
        queue_corr=queue_corr,
        tv_distance=tv,
        agg_mean_delay=agg_delay,
        agg_indep_delay=agg_indep,
        mean_queue=mean_queue,
    )


def _tv_distance_2d(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Total-variation distance between the empirical joint law of (q1, q2) and the
    product of its empirical marginals. Zero means the two coordinates are
    independent (product form holds).
    """
    qmax = int(max(q1.max(), q2.max())) + 1
    joint = np.zeros((qmax, qmax))
    for a, b in zip(q1, q2):
        joint[a, b] += 1
    joint /= joint.sum()
    m1 = joint.sum(axis=1)
    m2 = joint.sum(axis=0)
    prod = np.outer(m1, m2)
    return float(0.5 * np.abs(joint - prod).sum())
