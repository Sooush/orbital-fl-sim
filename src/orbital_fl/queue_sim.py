"""
Faithful discrete-event validation of the M/G/1 vacation staleness model.

The analytical model in queueing.py assumes gradients arrive as a Poisson(lam)
stream into a buffer that is served at rate mu whenever the satellite is in the
ready set R = {B > B_crit, C = 1} and is off otherwise. The on-off server path
comes from the real (B, I, C) Markov chain.

The original validation in queueing_analysis.py rolled the contact_gated policy,
under which gradients only arrive while the link is down (a_local = 1 - C). That
makes the realized arrival rate roughly lambda_g * P(C=0, B>B_crit) ~ 0.10, not
lambda_g = 0.5, so the analytical and empirical numbers were computed at
different loads. This module fixes that: it samples the server on-off path from
the actual environment and then injects an independent Poisson(lam) arrival
stream, exactly matching the assumptions behind the analytical formula. The
arrival rate is now a free knob, so we can sweep load and check that the
analytical curve tracks simulation across the whole stable range.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .env import EnvConfig, SatelliteEnv
from .policies import contact_gated


def server_ready_path(cfg: EnvConfig, horizon: int, seed: int = 0) -> np.ndarray:
    """
    Roll the environment under the contact_gated policy and record, for every
    slot, whether the satellite was in the ready set (C = 1 and B > B_crit).

    Returns a boolean array of length `horizon`. This is the on-off server
    sample path that drives the queue. We use contact_gated because that is the
    policy whose battery dynamics the Markov stationary analysis models, so the
    realized P_ready and off-period structure match markov.run_stationary_analysis.
    """
    env = SatelliteEnv(config=cfg)
    obs = env.reset(seed=seed)
    ready = np.zeros(horizon, dtype=bool)
    for t in range(horizon):
        ready[t] = (obs["C"] == 1) and (obs["B"] > cfg.B_crit)
        obs, _, _, _ = env.step(contact_gated(obs))
    return ready


def simulate_queue_on_path(
    ready_path: np.ndarray,
    lam: float,
    mu: int,
    seed: int = 0,
    warmup: int = 500,
) -> dict:
    """
    Run a single-server queue on a given on-off server path.

    At each slot: inject Poisson(lam) arrivals (independent of the server
    state, matching the analytical model), then if the server is ready drain
    up to mu gradients in FCFS order. Each drained gradient records its
    staleness (drain slot minus arrival slot).

    `warmup` slots are simulated but their drains are discarded so the reported
    statistics reflect steady state rather than the empty-queue transient.

    Returns mean/var staleness, the staleness samples, realized arrival and
    throughput rates, and the empirical busy/ready fraction.
    """
    rng = np.random.default_rng(seed)
    horizon = len(ready_path)
    queue: list[int] = []          # arrival slots, FCFS
    staleness: list[int] = []
    arrived = 0
    drained = 0

    for t in range(horizon):
        n_arr = int(rng.poisson(lam))
        arrived += n_arr
        queue.extend([t] * n_arr)

        if ready_path[t] and queue:
            k = min(mu, len(queue))
            for _ in range(k):
                a = queue.pop(0)
                if t >= warmup:
                    staleness.append(t - a)
            drained += k

    s = np.array(staleness, dtype=np.float64) if staleness else np.array([np.nan])
    return {
        "mean_staleness": float(np.mean(s)),
        "std_staleness": float(np.std(s)),
        "n_drained": len(staleness),
        "arrival_rate": arrived / horizon,
        "throughput": drained / horizon,
        "ready_frac": float(ready_path.mean()),
        "samples": s,
    }


def validate_staleness(
    cfg: EnvConfig,
    lam: float,
    mu: int,
    horizon: int = 20000,
    seeds: Optional[range] = None,
    warmup: int = 500,
) -> dict:
    """
    Multi-seed empirical mean staleness at arrival rate `lam`.

    For each seed we draw an independent server path and an independent arrival
    stream, then aggregate the per-seed mean staleness into a mean and a 95%
    confidence interval (normal approximation, 1.96 standard errors).
    """
    if seeds is None:
        seeds = range(20)
    seeds = list(seeds)
    per_seed = []
    for sd in seeds:
        ready = server_ready_path(cfg, horizon, seed=sd)
        out = simulate_queue_on_path(ready, lam, mu, seed=1000 + sd, warmup=warmup)
        per_seed.append(out["mean_staleness"])
    arr = np.array(per_seed, dtype=np.float64)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "lam": lam,
        "mean": mean,
        "ci95": 1.96 * sem,
        "sem": sem,
        "per_seed": arr,
        "n_seeds": len(arr),
    }
