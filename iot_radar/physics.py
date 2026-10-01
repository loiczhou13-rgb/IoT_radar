"""Physical constants and the monostatic radar range equation (link budget)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT: float = 299_792_458.0
"""Speed of light in vacuum (m/s)."""

BOLTZMANN: float = 1.380649e-23
"""Boltzmann constant (J/K)."""

T0: float = 290.0
"""Reference noise temperature (K)."""


def radar_range(
    params: dict[str, Any],
    wavelength: float,
    B_hz: float,
) -> float:
    """Evaluate the monostatic radar range equation."""
    P_tx = 1e-3 * 10.0 ** (params["P_tx_dBm"] / 10.0)
    G_tx = 10.0 ** (params["G_tx_dBi"] / 10.0)
    G_rx = 10.0 ** (params["G_rx_dBi"] / 10.0)
    sigma = params["sigma_m2"]
    F = 10.0 ** (params["NF_dB"] / 10.0)
    L = 10.0 ** (params["L_sys_dB"] / 10.0)
    SNR_min = 10.0 ** (params["SNR_min_dB"] / 10.0)

    numerator = P_tx * G_tx * G_rx * wavelength**2 * sigma
    denominator = (4.0 * np.pi)**3 * BOLTZMANN * T0 * B_hz * F * L * SNR_min
    return float((numerator / denominator) ** 0.25)


def compute_range(cfg: dict[str, Any]) -> tuple[float, float]:
    """Return ``(R_min, R_max)`` — pessimistic and optimistic ranges (m)."""
    bl = cfg["bilan_liaison"]
    f_c = float(cfg["sdr"]["f_c"])
    wavelength = SPEED_OF_LIGHT / f_c
    B_hz = float(bl["B_eff_hz"])

    R_max = radar_range(bl["optimiste"], wavelength, B_hz)
    R_min = radar_range(bl["pessimiste"], wavelength, B_hz)

    logger.info(
        "Portée effective — R_min = %.1f m (pire-cas) / R_max = %.1f m (optimiste) "
        "(B=%.3f Hz, λ=%.3f m)",
        R_min,
        R_max,
        B_hz,
        wavelength,
    )
    return R_min, R_max
