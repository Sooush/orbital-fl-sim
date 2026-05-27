# orbital-fl-sim

Simulator and analytical model for energy-aware queueing and bandwidth allocation in on-orbit federated learning.

Companion code for the EE 384S Spring 2026 project: *Energy-Aware Queueing and Bandwidth Allocation for On-Orbit Federated Learning* (Anbhuarasan, Khan, Gupta, Jonathan).

## Repository layout

```
orbital-fl-sim/
├── src/orbital_fl/
│   ├── env.py              # SatelliteEnv + EnvConfig — discrete-time (B,I,C,Q) MDP
│   ├── policies.py         # power_oblivious, contact_gated, power_weighted baselines
│   ├── markov.py           # stationary analysis: build P, solve pi*P=pi, P_ready, beta
│   ├── queueing.py         # analytical M/M/1 vacation model: E[T], E[L], batch penalty
│   ├── cmdp.py             # Lagrangian-relaxation CMDP: build_model, VI, pareto_sweep
│   └── __init__.py
├── scripts/
│   ├── sanity_check.py     # baseline policy rollout, figures/sanity_check.png
│   ├── queueing_analysis.py# Markov + queueing sweep, figures/queueing_analysis.png
│   └── cmdp_sweep.py       # CMDP Pareto sweep, figures/cmdp_pareto.png
├── figures/
│   ├── sanity_check.png
│   ├── queueing_analysis.png
│   └── cmdp_pareto.png
└── README.md
```

## Quickstart

```bash
# 1. Create a virtual environment (required — do not install globally)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install numpy matplotlib

# 3. Run baseline policy comparison
python scripts/sanity_check.py

# 4. Run Markov stationary analysis + queueing model
python scripts/queueing_analysis.py

# 5. Run CMDP Pareto sweep (bandwidth-allocation policy optimisation)
python scripts/cmdp_sweep.py
```

All scripts save figures to `figures/` and print a console summary.

## Model

### Simulator (`env.py`)

State: $S_t = (B_t, I_t, C_t, Q_t)$

- $B_t \in \{0,\dots,B_{\max}\}$: discrete battery level
- $I_t \in \{0,1\}$: illumination (0 = eclipse, 1 = sunlit)
- $C_t \in \{0,1\}$: ISL contact (0 = disconnected, 1 = connected)
- $Q_t \in \{0,\dots,Q_{\max}\}$: gradient queue length

Action: $a_t = (a_{\text{local}}, a_{\text{tx}}) \in [0,1]^2$

Energy (Harvest-Use-Store):
$$\Delta E = \eta I - P_{\text{tx}} a_{\text{tx}} C - P_{\text{gpu}} a_{\text{local}} - P_{\text{base}}$$
$$B_{t+1} = \text{clip}(B_t + \Delta E, 0, B_{\max})$$

Gradient arrivals are Poisson($\lambda_g \cdot a_{\text{local}}$), gated by $B_t > B_{\text{crit}}$.  
Service: up to $\lfloor a_{\text{tx}} \cdot \mu \rfloor$ gradients drained per slot when $C_t = 1$ and $B_t > B_{\text{crit}}$.

Each drained gradient emits a `DrainEvent(arrival_time, drain_time, sender_id, size)` — the schema consumed by the queueing analysis.

### Markov chain analysis (`markov.py`)

Builds the $(B_{\max}+1) \times 2 \times 2$ transition matrix under the **contact-gated policy** ($a_{\text{local}} = 1-C$, $a_{\text{tx}} = C$), which recovers the deterministic energy equation from the paper. Solves $\pi P = \pi$ by linear system and extracts:

- **$P_{\text{ready}}$** $= P(B > B_{\text{crit}},\, C=1)$: steady-state server uptime
- **$\beta$**: effective per-slot link-recovery probability (computed from the stationary flow into the ready set)

> **Calibration note.** The default `EnvConfig` has $\eta = 2.0$, which gives $\mathbb{E}[\Delta E] < 0$ in all states and causes the battery to drain to zero (degenerate case, $P_{\text{ready}} \to 0$). The analysis scripts use a calibrated config with $\eta = 4.0$, which yields a proper stationary battery distribution. Set $\eta$ so that $\eta \cdot P(\text{sunlit}) > P_{\text{tx}} \cdot P(C{=}1) + P_{\text{gpu}} \cdot P(C{=}0) + P_{\text{base}}$.

### Queueing model (`queueing.py`)

Each satellite's gradient buffer is an M/M/1 queue with a two-state on-off server. Server uptime is $P_{\text{ready}}$; off-periods are geometric with recovery probability $\beta$.

**Stability:** $\rho_{\text{eff}} = \lambda / (\mu \cdot P_{\text{ready}}) < 1$

**Mean staleness** (M/G/1 vacation decomposition, Doshi 1986):
$$\mathbb{E}[T] = \underbrace{\frac{1}{\mu - \lambda}}_{\text{M/M/1 sojourn}} + \underbrace{\frac{2-\beta}{2\beta}}_{\text{residual off-period}}$$

This is a conservative upper bound on the true staleness (overestimates because the multiple-vacation model is more pessimistic than the on-off server model).

**DiLoCo batch penalty** (M$^{[X]}$/M/1 correction):
$$\mathbb{E}[T]_{\text{batch}} = \mathbb{E}[T] + \frac{k-1}{2(\mu - \lambda)}$$

**Numerical results** (calibrated config, $\eta=4.0$, $\lambda=0.5$, $\mu=2$):

| Quantity | Value |
|---|---|
| $P_{\text{ready}}$ | 0.437 |
| $\beta$ | 0.049 |
| $\rho_{\text{eff}}$ | 0.572 |
| $\mathbb{E}[T]$ analytical | 20.7 slots |
| $\mathbb{E}[T]$ empirical (8000-step rollout) | 15.9 slots |

### CMDP solver (`cmdp.py`)

Formulates the bandwidth-allocation problem as a Constrained MDP over the full $(B, I, C, Q)$ state space.

**State:** $(B, I, C, Q)$ with $Q_{\max}^{\text{CMDP}} = 20$ — total 1,764 states.

**Actions:** $(a_{\text{local}}, a_{\text{tx}}) \in \{0,1\}^2$ — idle, train-only, tx-only, train+tx.

**Reward:** gradients drained per slot. **Cost:** $\mathbf{1}[B < B_{\text{crit}}]$.

**Lagrangian relaxation:** for each $\lambda$, solve:
$$V^\lambda(s) = \max_a\bigl[R(s,a) - \lambda\,\mathcal{C}(s) + \gamma\sum_{s'} P(s'|s,a)\,V^\lambda(s')\bigr]$$
via vectorised value iteration ($\gamma=0.99$, convergence < 1000 iterations).

**Pareto results** (calibrated config, 5000-step rollout):

| Policy | Throughput (grad/slot) | Cost fraction |
|---|---|---|
| power\_oblivious | 0.223 | 0.531 |
| contact\_gated   | 0.103 | 0.325 |
| power\_weighted  | 0.253 | 0.049 |
| CMDP ($\lambda=0$)    | 0.340 | 0.228 |
| CMDP ($\lambda=0.05$) | **0.323** | **0.000** |

The CMDP policy Pareto-dominates all heuristic baselines: it achieves 0.323 grad/slot at zero energy violations, vs. 0.253 for the best heuristic — a **28% throughput improvement** at strictly lower energy cost.

## Next steps

1. **Robustness sweeps.** Gradient compression ratio, ECC bit-flip noise, constellation size.
2. **Multi-satellite extension.** Add $N$ independent `SatelliteEnv` instances, verify product-form factorization empirically, and test with correlated eclipse schedules.
3. **JAX port.** Only if rollout speed becomes the bottleneck.

## Team

- Nimalan Anbhuarasan — Markov chain stationary analysis
- Sarosh Khan — CMDP, simulator (`env.py`, `policies.py`, `sanity_check.py`)
- Vidur Gupta — Queueing model and CMDP (`markov.py`, `queueing.py`, `cmdp.py`, `queueing_analysis.py`, `cmdp_sweep.py`)
- Jonathan — Report structure and slides
