"""
Shared calibrated configuration for all analysis scripts.

The default EnvConfig (eta=2.0) is energy-degenerate: every state has negative
expected energy balance, so the battery drains to zero and P_ready -> 0. The
calibrated config below uses eta=4.0, which gives a proper stationary battery
distribution. Calibration rule: choose eta so that
    eta * P(sunlit) - P_tx * P(C=1) - P_gpu * P(C=0) - P_base > 0.

Every script imports this single object so the queueing analysis, CMDP sweep,
constellation analysis, and robustness sweeps are all run at the same operating
point, with no silently-divergent copies.
"""

from __future__ import annotations

from .env import EnvConfig

ANALYSIS_CFG = EnvConfig(
    B_max=20, B_crit=4, B_init=10,
    eta=4.0,
    P_base=0.5, P_gpu=2.0, P_tx=1.5,
    p_eclipse_exit=0.028, p_eclipse_enter=0.018,
    p_link_acquire=0.10, p_link_lose=0.05,
    lambda_g=0.5, mu=2, Q_max=50,
    seed=42,
)
