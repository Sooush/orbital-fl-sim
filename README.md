# orbital-fl-sim

Simulator and analytical model for energy-aware queueing and bandwidth allocation in on-orbit federated learning.

Companion code for the EE 384S Spring 2026 project: *Energy-Aware Queueing and Bandwidth Allocation for On-Orbit Federated Learning* (Anbhuarasan, Khan, Gupta, Jonathan).

## Repository layout

```
orbital-fl-sim/
├── src/orbital_fl/
│   ├── env.py              # SatelliteEnv + EnvConfig, discrete-time (B,I,C,Q) MDP
│   ├── policies.py         # power_oblivious, contact_gated, power_weighted baselines
│   ├── markov.py           # stationary (B,I,C) solve + exact joint (Q,B,I,C) queue solve
│   ├── queueing.py         # analytical staleness: vacation, general off-period, discrete-time
│   ├── queue_sim.py        # discrete-event validation: Poisson arrivals on a sampled server path
│   ├── constellation.py    # N-satellite superposition: product form, merged rate, breakdown
│   ├── cmdp.py             # Lagrangian-relaxation CMDP: build_model, VI, pareto_sweep (multi-seed)
│   ├── analysis_config.py  # shared calibrated EnvConfig used by every script
│   └── __init__.py
├── scripts/
│   ├── sanity_check.py          # baseline policy rollout, figures/sanity_check.png
│   ├── queueing_analysis.py     # staleness models vs simulation, figures/queueing_analysis.png
│   ├── constellation_analysis.py# product form + eclipse breakdown, figures/constellation.png
│   ├── robustness_sweeps.py     # compression + fading limits, figures/robustness_sweeps.png
│   ├── cmdp_sweep.py            # CMDP Pareto sweep, figures/cmdp_pareto.png
│   └── run_all.py               # regenerate every figure and results/ log in one command
├── figures/
├── results/                # console logs from run_all.py (one .txt per script)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Create a virtual environment (required, do not install globally)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Regenerate every figure and console table in one command
python scripts/run_all.py

# ...or run an individual analysis:
python scripts/queueing_analysis.py      # staleness models vs simulation
python scripts/constellation_analysis.py # constellation product form + breakdown
python scripts/robustness_sweeps.py      # compression and fading limits
python scripts/cmdp_sweep.py             # CMDP Pareto sweep
```

All scripts save figures to `figures/` and print a console summary; `run_all.py` also
captures each script's output to `results/`.

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

Each drained gradient emits a `DrainEvent(arrival_time, drain_time, sender_id, size)`, the schema consumed by the queueing analysis.

### Markov chain analysis (`markov.py`)

Builds the $(B_{\max}+1) \times 2 \times 2$ transition matrix under the **contact-gated policy** ($a_{\text{local}} = 1-C$, $a_{\text{tx}} = C$), which recovers the deterministic energy equation from the paper. Solves $\pi P = \pi$ by linear system and extracts:

- **$P_{\text{ready}}$** $= P(B > B_{\text{crit}},\, C=1)$, the steady-state server uptime
- **$\beta$**, the effective per-slot link-recovery probability (computed from the stationary flow into the ready set)

It also provides the **exact joint $(Q, B, I, C)$ Markov-modulated queue solve** (`solve_queue_chain`) and the **off-period moments** (`offperiod_moments`). The gradient buffer is served only while the satellite is ready, so it is a discrete-time queue modulated by the energy-channel chain (a random-environment / quasi-birth-death process). Solving the joint chain by a sparse stationary solve gives the exact mean backlog and, via Little's law, the exact mean staleness.

> **Calibration note.** The default `EnvConfig` has $\eta = 2.0$, which gives $\mathbb{E}[\Delta E] < 0$ in all states and causes the battery to drain to zero (degenerate case, $P_{\text{ready}} \to 0$). The analysis scripts use a calibrated config (`analysis_config.ANALYSIS_CFG`) with $\eta = 4.0$, which yields a proper stationary battery distribution. Set $\eta$ so that $\eta \cdot P(\text{sunlit}) > P_{\text{tx}} \cdot P(C{=}1) + P_{\text{gpu}} \cdot P(C{=}0) + P_{\text{base}}$.

### Queueing model (`queueing.py`, `queue_sim.py`)

The gradient buffer is a discrete-time queue with a two-state on-off server. Server uptime is $P_{\text{ready}}$; the off-period is the sojourn in the not-ready set.

**Stability:** $\rho_{\text{eff}} = \lambda / (\mu \cdot P_{\text{ready}}) < 1$

**Closed-form vacation estimate** (M/G/1 multiple-vacation decomposition, Doshi 1986):
$$\mathbb{E}[T] = \frac{1}{\mu - \lambda} + \frac{2-\beta}{2\beta}$$
This assumes the off-period is geometric with rate $\beta$. The true off-period is two-phase (a fast contact-limited branch and a slow battery-limited branch) with squared coefficient of variation $\approx 2$, so the geometric estimate is unreliable: it stays near 20 slots regardless of load while the true staleness ranges from 19 to 83 slots over $\rho_{\text{eff}} \in [0.1, 0.8]$. The **exact Markov-modulated solve** matches a Poisson-arrival discrete-event rollout across the whole load range (`queue_sim.py` provides the load-matched validation, replacing an earlier rollout whose realized rate was about 0.10 grad/slot rather than the model's $\lambda$).

**DiLoCo batch penalty** (M$^{[X]}$/M/1 correction):
$$\mathbb{E}[T]_{\text{batch}} = \mathbb{E}[T] + \frac{k-1}{2(\mu - \lambda)}$$

**Numerical results** (calibrated config, $\eta=4.0$, $\lambda=0.5$, $\mu=2$, 16-seed rollout):

| Quantity | Value |
|---|---|
| $P_{\text{ready}}$ | 0.437 |
| $\beta$ | 0.049 |
| $\rho_{\text{eff}}$ | 0.572 |
| off-period $\mathbb{E}[V]$, $\mathrm{cv}^2$ | 20.5 slots, 1.98 |
| $\mathbb{E}[T]$ vacation (geometric $\beta$) | 20.7 slots |
| $\mathbb{E}[T]$ exact Markov-modulated | 38.9 slots |
| $\mathbb{E}[T]$ simulation | 39.1 $\pm$ 2.9 slots |

### Constellation analysis (`constellation.py`)

Runs $N$ independent satellites plus an aggregator (the discrete-time superposition of on/off sources). Confirms the merged arrival rate equals $N\lambda$ (flow balance) and that the per-satellite backlogs factor (product form) when the chains are independent. A correlation knob $\rho_{\text{corr}}$ shares the eclipse chain across the plane; as it grows, the per-satellite queues become positively correlated and aggregation delay climbs above the independent baseline, which is the regime where product form fails. The aggregator delay exceeds the naive $1/(\mu_{\text{agg}} - N\lambda)$ M/M/1 prediction even at independence, because the on/off departure streams are bursty rather than Poisson.

### Robustness sweeps (`robustness_sweeps.py`)

Sweeps gradient compression (effective service rate $\mu_{\text{eff}}$) and ISL fading (link-loss probability). Compression has diminishing returns toward an instant-service floor set by the off-period wait; the fading limit is the link-loss rate at which $P_{\text{ready}}$ drops to $\lambda/\mu$ and the queue goes unstable ($p_{\text{loss}} \approx 0.16$ at the calibrated point).

### CMDP solver (`cmdp.py`)

Formulates the bandwidth-allocation problem as a Constrained MDP over the full $(B, I, C, Q)$ state space.

**State:** $(B, I, C, Q)$ with $Q_{\max}^{\text{CMDP}} = 20$, total 1,764 states.

**Actions:** $(a_{\text{local}}, a_{\text{tx}}) \in \{0,1\}^2$, i.e. idle, train-only, tx-only, train+tx.

**Reward:** gradients drained per slot. **Cost:** $\mathbf{1}[B < B_{\text{crit}}]$.

**Lagrangian relaxation:** for each multiplier $\theta$, solve:
$$V^\theta(s) = \max_a\bigl[R(s,a) - \theta\,\mathcal{C}(s) + \gamma\sum_{s'} P(s'|s,a)\,V^\theta(s')\bigr]$$
via vectorised value iteration ($\gamma=0.99$, convergence < 1000 iterations).

**Pareto results** (calibrated config, 5000-step rollout, 20 seeds, 95% CI):

| Policy | Throughput (grad/slot) | Cost fraction |
|---|---|---|
| power\_oblivious | 0.246 $\pm$ 0.011 | 0.479 $\pm$ 0.020 |
| contact\_gated   | 0.109 $\pm$ 0.005 | 0.317 $\pm$ 0.019 |
| power\_weighted  | 0.284 $\pm$ 0.008 | 0.037 $\pm$ 0.010 |
| CMDP ($\theta=0$)    | 0.328 $\pm$ 0.007 | 0.275 $\pm$ 0.016 |
| CMDP ($\theta=0.05$) | **0.326 $\pm$ 0.010** | **0.000 $\pm$ 0.000** |

The CMDP at $\theta \geq 0.05$ holds zero energy-floor violations across all 20 seeds (worst-seed cost 0.000), while the best heuristic still violates (worst-seed 0.081). At zero violations the CMDP delivers a **14.7% throughput gain** over the best heuristic.

The CMDP policy Pareto-dominates all heuristic baselines: it achieves 0.323 grad/slot at zero energy violations, vs. 0.253 for the best heuristic — a **28% throughput improvement** at strictly lower energy cost.

## Next steps

1. **ECC bit-flip noise.** Add a gradient-corruption channel and measure its effect on useful throughput (the compression and constellation-size sweeps are now implemented).
2. **Matrix-geometric form.** Replace the truncated sparse solve with an explicit matrix-geometric $R$ for the joint queue, removing the truncation parameter.
3. **JAX port.** Only if rollout speed becomes the bottleneck.

## Team

- Nimalan Anbhuarasan, Markov chain stationary analysis
- Sarosh Khan, CMDP and simulator (`env.py`, `policies.py`, `sanity_check.py`)
- Vidur Gupta, queueing and bandwidth allocation (`markov.py`, `queueing.py`, `queue_sim.py`, `constellation.py`, `cmdp.py`, and the analysis scripts)
- Jonathan, report structure and slides
