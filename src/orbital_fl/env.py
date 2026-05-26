"""
Single-satellite environment for the Federated Orbital Edge Mesh project.

State:  S_t = (B_t, I_t, C_t, Q_t)
    B_t in {0, ..., B_max}       discrete battery level
    I_t in {0, 1}                illumination (0=eclipse, 1=sunlit)
    C_t in {0, 1}                ISL contact (0=no link, 1=connected)
    Q_t in {0, ..., Q_max}       gradient queue length

Action: a_t = (a_local, a_tx) in [0,1]^2
    a_local : fraction of slot spent training (consumes P_gpu * a_local,
                                                drives gradient arrivals)
    a_tx    : fraction of bandwidth claimed   (consumes P_tx * a_tx,
                                                drives gradient drains, gated by C_t)

Energy dynamics (HUS):
    dE = eta * I  -  P_tx * a_tx * C  -  P_gpu * a_local  -  P_base
    B_{t+1} = clip(B_t + dE, 0, B_max)

Gradient dynamics:
    arrival: Poisson(lambda_g * a_local), gated by had_energy
    service: min(Q, floor(a_tx * C * mu))  gradients drained per step (deterministic)

Reward = number of gradients drained this step (real throughput).

Per-gradient events emitted on drain include:
    arrival_time, drain_time, sender_id, size
That's the schema Vidur's queueing-network model consumes.

Time and energy are in abstract units. Calibrate later once Nimalan's
stationary analysis tells us what regime is interesting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np


# ---------- gradient record ----------------------------------------------- #

@dataclass
class Gradient:
    """One pending gradient update sitting in the queue."""
    arrival_time: int   # step at which it was produced by local SGD
    sender_id: int      # which satellite produced it
    size: float = 1.0   # in abstract units; sweep this for compression studies


@dataclass
class DrainEvent:
    """Emitted when a gradient is successfully pushed over the ISL.

    This is the per-event schema we promised Vidur's queueing model.
    From these four fields he can compute staleness, throughput,
    inter-departure times, anything.
    """
    arrival_time: int
    drain_time: int
    sender_id: int
    size: float

    @property
    def staleness(self) -> int:
        return self.drain_time - self.arrival_time


# ---------- configuration -------------------------------------------------- #

@dataclass
class EnvConfig:
    # Battery
    B_max: int = 20
    B_init: int = 10
    B_crit: int = 4

    # Power coefficients (abstract units per timestep)
    eta: float = 2.0
    P_base: float = 0.5
    P_gpu: float = 2.0
    P_tx: float = 1.5

    # Illumination Markov chain (two-state, sticky)
    p_eclipse_exit: float = 0.028   # P(I=1 | I=0)
    p_eclipse_enter: float = 0.018  # P(I=0 | I=1)

    # ISL contact Markov chain (Gilbert-Elliott)
    p_link_acquire: float = 0.10    # P(C=1 | C=0)
    p_link_lose:    float = 0.05    # P(C=0 | C=1)

    # Gradient queue dynamics
    lambda_g: float = 0.5           # Poisson arrival rate per step at a_local=1
    mu: int = 2                     # max gradients drained per step at a_tx=1, C=1
    gradient_size: float = 1.0      # abstract units per gradient
    Q_max: int = 50                 # buffer cap (prevents unbounded growth)

    # Initial state
    I_init: int = 1
    C_init: int = 1

    # Reproducibility
    seed: Optional[int] = None


# ---------- environment ---------------------------------------------------- #

class SatelliteEnv:
    """
    Minimal Gym-style env for one satellite with energy and gradient-queue
    dynamics. satellite_id / neighbor_ids are constructor args so scaling to N
    satellites later doesn't require rewriting the class.
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        satellite_id: int = 0,
        neighbor_ids: Optional[list[int]] = None,
    ):
        self.cfg = config if config is not None else EnvConfig()
        self.satellite_id = satellite_id
        self.neighbor_ids = neighbor_ids if neighbor_ids is not None else []
        self.rng = np.random.default_rng(self.cfg.seed)

        # state - set by reset()
        self.B: int = 0
        self.I: int = 0
        self.C: int = 0
        self.t: int = 0
        self.queue: deque[Gradient] = deque()

        # cumulative diagnostics
        self.total_arrived: int = 0
        self.total_drained: int = 0
        self.total_dropped: int = 0     # arrivals lost to Q_max overflow

        self.reset()

    # ----- core API ----- #

    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.B = self.cfg.B_init
        self.I = self.cfg.I_init
        self.C = self.cfg.C_init
        self.t = 0
        self.queue = deque()
        self.total_arrived = 0
        self.total_drained = 0
        self.total_dropped = 0
        return self._obs()

    def step(self, action: Tuple[float, float]) -> Tuple[dict, float, bool, dict]:
        """
        Advance one timestep.

        Order of operations within a step:
            1. Energy update (under current I, C).
            2. Gradient arrivals (driven by a_local).
            3. Gradient drains (driven by a_tx * C, capped by mu and Q).
            4. Reward and cost computed from drains and battery state.
            5. Exogenous transitions of I and C.
            6. Commit B, I, C, t.

        Returns (obs, reward, done, info).
        """
        a_local, a_tx = action
        a_local = float(np.clip(a_local, 0.0, 1.0))
        a_tx    = float(np.clip(a_tx,    0.0, 1.0))

        # 1. Energy update
        dE = (
            self.cfg.eta * self.I
            - self.cfg.P_tx * a_tx * self.C
            - self.cfg.P_gpu * a_local
            - self.cfg.P_base
        )
        B_next = int(np.clip(round(self.B + dE), 0, self.cfg.B_max))
        had_energy = self.B > self.cfg.B_crit

        # 2. Arrivals: Poisson(lambda_g * a_local), gated by had_energy.
        # Poisson chosen for consistency with Jackson-network assumptions in
        # Vidur's downstream queueing analysis.
        arrived_this_step = 0
        if had_energy:
            n_arrivals = int(self.rng.poisson(self.cfg.lambda_g * a_local))
            space_left = self.cfg.Q_max - len(self.queue)
            accepted = min(n_arrivals, space_left)
            dropped  = n_arrivals - accepted
            for _ in range(accepted):
                self.queue.append(Gradient(
                    arrival_time=self.t,
                    sender_id=self.satellite_id,
                    size=self.cfg.gradient_size,
                ))
            arrived_this_step = accepted
            self.total_arrived += accepted
            self.total_dropped += dropped

        # 3. Service / drains
        drained_events: List[DrainEvent] = []
        if had_energy and self.C == 1:
            capacity = int(np.floor(a_tx * self.cfg.mu))
            to_drain = min(capacity, len(self.queue))
            for _ in range(to_drain):
                g = self.queue.popleft()
                drained_events.append(DrainEvent(
                    arrival_time=g.arrival_time,
                    drain_time=self.t,
                    sender_id=g.sender_id,
                    size=g.size,
                ))
            self.total_drained += to_drain

        # 4. Reward = real throughput; cost = energy-floor violation
        reward = float(len(drained_events))
        cost = 1.0 if self.B < self.cfg.B_crit else 0.0

        info = {
            "t": self.t,
            "B": self.B,
            "I": self.I,
            "C": self.C,
            "Q": len(self.queue),
            "a_local": a_local,
            "a_tx": a_tx,
            "dE": dE,
            "arrived_this_step": arrived_this_step,
            "drained_count": len(drained_events),
            "drained_events": drained_events,
            "cost": cost,
            "battery_floor_hit": (self.B + dE) < 0,
        }

        # 5. Exogenous transitions
        I_next = self._transition_two_state(
            self.I, self.cfg.p_eclipse_exit, self.cfg.p_eclipse_enter
        )
        C_next = self._transition_two_state(
            self.C, self.cfg.p_link_acquire, self.cfg.p_link_lose
        )

        # 6. Commit
        self.B, self.I, self.C = B_next, I_next, C_next
        self.t += 1

        done = False
        return self._obs(), reward, done, info

    # ----- helpers ----- #

    def _transition_two_state(self, s: int, p_01: float, p_10: float) -> int:
        u = self.rng.random()
        if s == 0:
            return 1 if u < p_01 else 0
        else:
            return 0 if u < p_10 else 1

    def _obs(self) -> dict:
        return {"B": self.B, "I": self.I, "C": self.C, "Q": len(self.queue), "t": self.t}

    # ----- introspection ----- #

    @property
    def is_ready(self) -> bool:
        return self.C == 1 and self.B > self.cfg.B_crit