"""
Single-satellite environment for the Federated Orbital Edge Mesh.

State:  S_t = (B_t, I_t, C_t)
    B_t in {0, ..., B_max}     discrete battery level
    I_t in {0, 1}              illumination (0=eclipse, 1=sunlit)
    C_t in {0, 1}              ISL contact (0=no link, 1=connected)

Action: a_t = (a_local, a_tx) in [0,1]^2
    a_local : fraction of slot spent training (consumes P_gpu * a_local)
    a_tx    : fraction of bandwidth claimed (consumes P_tx * a_tx, gated by C_t)

Energy dynamics (HUS):
    dE = eta * I  -  P_tx * a_tx * C  -  P_gpu * a_local  -  P_base
    B_{t+1} = clip(B_t + dE, 0, B_max)

Time and energy are in abstract units. Calibrate later once Nimalan's
stationary analysis tells us what regime is interesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------- configuration -------------------------------------------------- #

@dataclass
class EnvConfig:
    # Battery
    B_max: int = 20                # discrete battery buckets
    B_init: int = 10               # initial battery level
    B_crit: int = 4                # below this, satellite is considered "not ready"

    # Power coefficients (abstract units per timestep)
    eta: float = 2.0               # solar harvest rate when sunlit
    P_base: float = 0.5            # baseline avionics draw (always on)
    P_gpu: float = 2.0             # GPU draw at full local-step utilization
    P_tx: float = 1.5              # ISL transmit draw at full bandwidth claim

    # Illumination Markov chain (two-state)
    # State 0 = eclipse, State 1 = sunlit. Sticky transitions.
    # If orbital period ~ T steps and eclipse fraction ~ 0.4,
    # mean dwell times ~ 0.4T (eclipse) and 0.6T (sunlit).
    # For T = 90 steps: p_eclipse_exit = 1/(0.4*90) ~ 0.028 etc.
    p_eclipse_exit: float = 0.028   # P(I=1 | I=0)
    p_eclipse_enter: float = 0.018  # P(I=0 | I=1)

    # ISL contact Markov chain (Gilbert-Elliott)
    # State 0 = no link, State 1 = connected.
    p_link_acquire: float = 0.10    # P(C=1 | C=0)
    p_link_lose:    float = 0.05    # P(C=0 | C=1)

    # Initial state
    I_init: int = 1
    C_init: int = 1

    # Reproducibility
    seed: Optional[int] = None


# ---------- environment ---------------------------------------------------- #

class SatelliteEnv:
    """
    A minimal Gym-style env for one satellite.

    Designed so that satellite_id and neighbor_ids can be set at construction
    even though only one satellite exists today, so we can scale to N without
    rewriting the class.
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

        # state variables - set by reset()
        self.B: int = 0
        self.I: int = 0
        self.C: int = 0
        self.t: int = 0

        self.reset()

    # ----- core API ----- #

    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.B = self.cfg.B_init
        self.I = self.cfg.I_init
        self.C = self.cfg.C_init
        self.t = 0
        return self._obs()

    def step(self, action: Tuple[float, float]) -> Tuple[dict, float, bool, dict]:
        """
        Advance one timestep.

        Returns (obs, reward, done, info).
        Reward is a placeholder for now: gradients-pushed minus battery-floor-violation.
        We can swap this for the proper CMDP reward later.
        """
        a_local, a_tx = action
        a_local = float(np.clip(a_local, 0.0, 1.0))
        a_tx    = float(np.clip(a_tx,    0.0, 1.0))

        # 1. Energy update (HUS, evaluated under *current* I and C)
        dE = (
            self.cfg.eta * self.I
            - self.cfg.P_tx * a_tx * self.C
            - self.cfg.P_gpu * a_local
            - self.cfg.P_base
        )
        B_next = int(np.clip(round(self.B + dE), 0, self.cfg.B_max))

        # 2. Exogenous Markov transitions for I and C
        I_next = self._transition_two_state(
            self.I, self.cfg.p_eclipse_exit, self.cfg.p_eclipse_enter
        )
        C_next = self._transition_two_state(
            self.C, self.cfg.p_link_acquire, self.cfg.p_link_lose
        )

        # 3. Compute reward and info before committing the transition
        # "Productive work" this slot: local training that actually happened,
        # plus successful gradient push (only counts if connected AND had energy).
        had_energy = self.B > self.cfg.B_crit
        local_work = a_local if had_energy else 0.0
        tx_work = a_tx * self.C if had_energy else 0.0
        reward = local_work + 2.0 * tx_work   # tx weighted higher; tuneable

        info = {
            "t": self.t,
            "B": self.B,
            "I": self.I,
            "C": self.C,
            "a_local": a_local,
            "a_tx": a_tx,
            "dE": dE,
            "local_work": local_work,
            "tx_work": tx_work,
            "battery_floor_hit": (self.B + dE) < 0,
        }

        # 4. Commit
        self.B, self.I, self.C = B_next, I_next, C_next
        self.t += 1

        done = False   # we run for a fixed horizon externally
        return self._obs(), reward, done, info

    # ----- helpers ----- #

    def _transition_two_state(self, s: int, p_01: float, p_10: float) -> int:
        """Bernoulli flip on a two-state chain."""
        u = self.rng.random()
        if s == 0:
            return 1 if u < p_01 else 0
        else:
            return 0 if u < p_10 else 1

    def _obs(self) -> dict:
        return {"B": self.B, "I": self.I, "C": self.C, "t": self.t}

    # ----- introspection ----- #

    @property
    def is_ready(self) -> bool:
        """P_ready proxy at the instantaneous level: connected and above B_crit."""
        return self.C == 1 and self.B > self.cfg.B_crit
