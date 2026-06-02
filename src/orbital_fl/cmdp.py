"""
Lagrangian-relaxation CMDP for on-orbit bandwidth allocation.

State  : (B, I, C, Q) -- battery, illumination, contact, queue length.
Actions: (a_local, a_tx) in {0,1}^2 -- four binary choices.
Reward : gradients drained per slot.
Cost   : 1[B < B_crit] -- energy-floor indicator (action-independent).

For each multiplier lam solve
    V(s) = max_a [R(s,a) - lam*Cost(s) + gamma * sum_s' P(s'|s,a)*V(s')]
via vectorised value iteration. Sweeping lam from 0 to inf traces the
throughput / energy-floor-fraction Pareto frontier.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from .env import EnvConfig, SatelliteEnv


ACTIONS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
ACT_LABELS = ["idle", "train", "tx", "train+tx"]
N_ACT = 4
_MAX_ARR   = 8                       # Poisson truncation (tail < 1e-6 for lam_g <= 2)
_MAX_TRANS = 4 * (_MAX_ARR + 1)      # upper bound on unique next-states per (s,a)


@dataclass
class CMDPConfig:
    Q_max:    int   = 20     # queue cap for DP (env Q_max may be larger)
    gamma:    float = 0.99
    tol:      float = 0.01   # ||V_{n+1} - V_n||_inf stopping criterion
    max_iter: int   = 1000


# ---------- helpers --------------------------------------------------------- #

def _pmf(lam: float) -> np.ndarray:
    """Truncated Poisson PMF on {0,..,_MAX_ARR}; tail probability lumped into last bin."""
    p = np.zeros(_MAX_ARR + 1)
    p[0] = np.exp(-lam)
    for k in range(1, _MAX_ARR + 1):
        p[k] = p[k - 1] * lam / k
    p[_MAX_ARR] += max(0.0, 1.0 - p.sum())
    return p


def _enc(b: int, i: int, c: int, q: int, qn: int) -> int:
    return b * 4 * qn + i * 2 * qn + c * qn + q


# ---------- model construction ---------------------------------------------- #

def build_model(env_cfg: EnvConfig, cmdp_cfg: CMDPConfig):
    """
    Construct immediate reward/cost arrays and padded transition arrays.

    Returns
    -------
    R      : (n_s, N_ACT) float64 -- expected immediate reward
    Cost   : (n_s, N_ACT) float64 -- immediate cost (= 1[B < B_crit])
    T_idx  : (n_s*N_ACT, _MAX_TRANS) int32  -- next-state indices (padded with 0)
    T_prob : (n_s*N_ACT, _MAX_TRANS) float64 -- transition probs (0 for padding)
    n_s    : int -- total state count
    """
    B_max = env_cfg.B_max
    Q_max = cmdp_cfg.Q_max
    qn    = Q_max + 1
    n_s   = (B_max + 1) * 4 * qn

    pI = np.array([[1 - env_cfg.p_eclipse_exit, env_cfg.p_eclipse_exit],
                   [env_cfg.p_eclipse_enter,     1 - env_cfg.p_eclipse_enter]])
    pC = np.array([[1 - env_cfg.p_link_acquire,  env_cfg.p_link_acquire],
                   [env_cfg.p_link_lose,          1 - env_cfg.p_link_lose]])

    pmf1 = _pmf(env_cfg.lambda_g)
    pmf0 = np.zeros(_MAX_ARR + 1); pmf0[0] = 1.0

    R      = np.zeros((n_s, N_ACT))
    Cost   = np.zeros((n_s, N_ACT))
    T_idx  = np.zeros((n_s * N_ACT, _MAX_TRANS), dtype=np.int32)
    T_prob = np.zeros((n_s * N_ACT, _MAX_TRANS))

    for b in range(B_max + 1):
        he = b > env_cfg.B_crit
        cv = 1.0 if b < env_cfg.B_crit else 0.0
        for i in range(2):
            for c in range(2):
                for q in range(qn):
                    s = _enc(b, i, c, q, qn)
                    for a_idx, (al, at) in enumerate(ACTIONS):
                        dE  = (env_cfg.eta * i
                               - env_cfg.P_tx  * at * c
                               - env_cfg.P_gpu * al
                               - env_cfg.P_base)
                        bn  = int(np.clip(round(b + dE), 0, B_max))
                        dr  = min(q, int(at * env_cfg.mu)) if (he and c == 1) else 0
                        qa  = q - dr

                        R[s, a_idx]    = float(dr)
                        Cost[s, a_idx] = cv

                        pmf = pmf1 if (he and al > 0) else pmf0
                        td  = defaultdict(float)
                        for k, pk in enumerate(pmf):
                            if pk < 1e-14:
                                continue
                            q2 = int(np.clip(qa + k, 0, Q_max))
                            for i2 in range(2):
                                for c2 in range(2):
                                    td[_enc(bn, i2, c2, q2, qn)] += (
                                        pk * pI[i, i2] * pC[c, c2])

                        row = s * N_ACT + a_idx
                        for col, (sn, pr) in enumerate(td.items()):
                            T_idx[row, col]  = sn
                            T_prob[row, col] = pr

    return R, Cost, T_idx, T_prob, n_s


# ---------- value iteration ------------------------------------------------- #

def value_iteration(R, Cost, T_idx, T_prob, n_s,
                    lam: float, gamma: float, tol: float, max_iter: int):
    """Vectorised discounted VI for a fixed Lagrange multiplier lam."""
    R_lam = (R - lam * Cost).ravel()
    V = np.zeros(n_s)
    for _ in range(max_iter):
        EV    = (T_prob * V[T_idx]).sum(axis=1)
        V_new = (R_lam + gamma * EV).reshape(n_s, N_ACT).max(axis=1)
        if np.abs(V_new - V).max() < tol:
            return V_new
        V = V_new
    return V


def extract_policy(R, Cost, T_idx, T_prob, V, n_s, lam: float, gamma: float):
    """Greedy policy: pi[s] in {0,1,2,3} indexing ACTIONS."""
    R_lam = (R - lam * Cost).ravel()
    EV    = (T_prob * V[T_idx]).sum(axis=1)
    return (R_lam + gamma * EV).reshape(n_s, N_ACT).argmax(axis=1)


# ---------- evaluation ------------------------------------------------------ #

def rollout_policy(env_cfg: EnvConfig, pi, Q_max: int,
                   horizon: int = 5000, seed: int = 0):
    """Evaluate a flat policy array on SatelliteEnv. Returns (throughput, cost_frac)."""
    env = SatelliteEnv(config=env_cfg)
    obs = env.reset(seed=seed)
    qn  = Q_max + 1
    tr, tc = 0.0, 0.0
    for _ in range(horizon):
        b = obs["B"]; i = obs["I"]; c = obs["C"]
        q = min(obs["Q"], Q_max)
        s = _enc(b, i, c, q, qn)
        al, at = ACTIONS[int(pi[s])]
        obs, r, _, info = env.step((al, at))
        tr += r
        tc += info["cost"]
    return tr / horizon, tc / horizon


def rollout_policy_ci(env_cfg: EnvConfig, pi, Q_max: int,
                      horizon: int = 5000, seeds=range(20)):
    """
    Multi-seed evaluation of a flat policy. Runs one rollout per seed and
    aggregates throughput and energy-floor cost into means with 95% confidence
    intervals (normal approximation). Also reports the worst-case cost fraction
    over the seed set, so a "zero violations" claim is shown to hold across seeds
    rather than at a single lucky seed.
    """
    seeds = list(seeds)
    thr = np.empty(len(seeds))
    cost = np.empty(len(seeds))
    for k, sd in enumerate(seeds):
        thr[k], cost[k] = rollout_policy(env_cfg, pi, Q_max, horizon, seed=sd)
    def ci(x):
        return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return {
        "thr_mean": float(thr.mean()), "thr_ci": float(ci(thr)),
        "cost_mean": float(cost.mean()), "cost_ci": float(ci(cost)),
        "cost_max": float(cost.max()), "n_seeds": len(seeds),
    }


# ---------- Pareto sweep ---------------------------------------------------- #

def pareto_sweep(env_cfg: EnvConfig, cmdp_cfg: CMDPConfig,
                 lam_values, horizon: int = 5000, seeds=range(20)):
    """
    Build the model once, then for each lam solve value iteration and evaluate
    the greedy policy over a set of seeds.

    Returns ndarray shape (len(lam_values), 5):
        [lam, thr_mean, thr_ci, cost_mean, cost_ci]
    so every Pareto point carries a 95% confidence interval.
    """
    print("Building CMDP model...", flush=True)
    R, Cost, T_idx, T_prob, n_s = build_model(env_cfg, cmdp_cfg)
    print(f"  states={n_s}  actions={N_ACT}  transition_cols={_MAX_TRANS}",
          flush=True)

    rows = []
    for lam in lam_values:
        V  = value_iteration(R, Cost, T_idx, T_prob, n_s,
                             lam, cmdp_cfg.gamma, cmdp_cfg.tol, cmdp_cfg.max_iter)
        pi = extract_policy(R, Cost, T_idx, T_prob, V, n_s, lam, cmdp_cfg.gamma)
        ev = rollout_policy_ci(env_cfg, pi, cmdp_cfg.Q_max, horizon, seeds)
        rows.append((lam, ev["thr_mean"], ev["thr_ci"], ev["cost_mean"], ev["cost_ci"]))
        print(f"  lam={lam:6.2f}  thr={ev['thr_mean']:.3f}+/-{ev['thr_ci']:.3f}  "
              f"cost={ev['cost_mean']:.3f}+/-{ev['cost_ci']:.3f}  cost_max={ev['cost_max']:.3f}",
              flush=True)

    return np.array(rows)
