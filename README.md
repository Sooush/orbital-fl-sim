# orbital-fl-sim

Simulator for energy-aware queueing and bandwidth allocation in on-orbit federated learning.

Companion code for the EE 384S Spring 2026 project: *Energy-Aware Queueing and
Bandwidth Allocation for On-Orbit Federated Learning* (Anbhuarasan, Khan, Gupta).

## What's here (v0)

A single-satellite Markov environment that exercises the full state space from
the proposal: battery, illumination, ISL contact. Three baseline policies and a
sanity-check script that produces the first comparison plot.

```
orbital-fl-sim/
├── src/orbital_fl/
│   ├── env.py          # SatelliteEnv + EnvConfig
│   ├── policies.py     # power_oblivious, contact_gated, power_weighted
│   └── __init__.py
├── scripts/
│   └── sanity_check.py # rolls out each policy, saves figures/sanity_check.png
├── figures/
└── README.md
```

## Quickstart

```bash
pip install numpy matplotlib
python scripts/sanity_check.py
```

Output: console summary of avg reward / battery-dead fraction per policy, plus
`figures/sanity_check.png`.

## Model

State: $S_t = (B_t, I_t, C_t)$.

Action: $a_t = (a_{\text{local}}, a_{\text{tx}}) \in [0,1]^2$.

Energy (Harvest-Use-Store):
$$\Delta E = \eta \cdot I - P_{\text{tx}} \cdot a_{\text{tx}} \cdot C - P_{\text{gpu}} \cdot a_{\text{local}} - P_{\text{base}}$$
$$B_{t+1} = \text{clip}(B_t + \Delta E, 0, B_{\max})$$

Illumination and contact are independent two-state Markov chains, parameterized
in `EnvConfig`. Units are abstract; calibrate once the stationary analysis tells
us what regime is interesting.

The `contact_gated` baseline recovers the original deterministic energy equation
in the proposal exactly (`a_local = 1 - C`, `a_tx = C`), which is useful as a
sanity check against Nimalan's stationary analysis.

## Next steps

In rough priority order:

1. **Gradient queue.** Replace the hand-tuned reward proxy with a real buffer:
   gradients arrive at some rate, drain at `a_tx * C * mu`, reward = drained.
   This is also the interface point with Vidur's queueing-network model.
2. **Proper constraint signal.** Separate cost stream
   $c_t = \mathbb{1}[B_t < B_{\text{crit}}]$ so the CMDP solver treats energy
   neutrality as a hard constraint, not reward shaping.
3. **Lagrangian-relaxation solver.** Sweep $\lambda$, value-iterate on the
   discretized state space, plot the Pareto frontier of throughput vs. energy
   deficit. This is the headline figure.
4. **Robustness sweeps.** Gradient compression ratio, constellation size (once
   multi-satellite is in), ECC bit-flip noise.
5. **JAX port.** Only if rollout speed becomes the bottleneck.

## Team

- Nimalan Anbhuarasan — Markov chain stationary analysis
- Sarosh Khan — CMDP, simulator, experiments (this repo)
- Vidur Gupta — Queueing-network model of ISL aggregation
