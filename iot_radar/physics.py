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
    P_tx = 1e-3 * 10.0 ** (params["tx_power_dbm"] / 10.0)
    G_tx = 10.0 ** (params["tx_antenna_gain_dbi"] / 10.0)
    G_rx = 10.0 ** (params["rx_antenna_gain_dbi"] / 10.0)
    sigma = params["radar_cross_section_m2"]
    F = 10.0 ** (params["noise_figure_db"] / 10.0)
    L = 10.0 ** (params["system_losses_db"] / 10.0)
    SNR_min = 10.0 ** (params["min_snr_db"] / 10.0)

    numerator = P_tx * G_tx * G_rx * wavelength**2 * sigma
    denominator = (4.0 * np.pi)**3 * BOLTZMANN * T0 * B_hz * F * L * SNR_min
    return float((numerator / denominator) ** 0.25)


def compute_range(cfg: dict[str, Any]) -> tuple[float, float]:
    """Return ``(R_min, R_max)`` — pessimistic and optimistic ranges (m)."""
    bl = cfg["link_budget"]
    f_c = float(cfg["sdr"]["center_frequency_hz"])
    wavelength = SPEED_OF_LIGHT / f_c
    B_hz = float(bl["noise_bandwidth_hz"])

    R_max = radar_range(bl["optimistic"], wavelength, B_hz)
    R_min = radar_range(bl["pessimistic"], wavelength, B_hz)

    logger.info(
        "Portée effective — R_min = %.1f m (pire-cas) / R_max = %.1f m (optimiste) "
        "(B=%.3f Hz, λ=%.3f m)",
        R_min,
        R_max,
        B_hz,
        wavelength,
    )
    return R_min, R_max
