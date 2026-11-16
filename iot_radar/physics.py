"""Physical constants and the radar range equation (range shown on the dashboard)."""

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


def radar_range_m(
    scenario: dict[str, Any],
    wavelength_m: float,
    noise_bandwidth_hz: float,
) -> float:
    """Range (m) at which the echo reaches ``min_snr_db`` (monostatic radar equation).

    ``R = [P_t G_t G_r λ² σ / ((4π)³ k T0 B F L SNR_min)]^(1/4)``; *scenario* is
    one scenario of the ``link_budget`` section.
    """
    tx_power_w = 1e-3 * 10.0 ** (scenario["tx_power_dbm"] / 10.0)
    tx_gain = 10.0 ** (scenario["tx_antenna_gain_dbi"] / 10.0)
    rx_gain = 10.0 ** (scenario["rx_antenna_gain_dbi"] / 10.0)
    cross_section_m2 = scenario["radar_cross_section_m2"]
    noise_factor = 10.0 ** (scenario["noise_figure_db"] / 10.0)
    losses = 10.0 ** (scenario["system_losses_db"] / 10.0)
    min_snr = 10.0 ** (scenario["min_snr_db"] / 10.0)

    numerator = tx_power_w * tx_gain * rx_gain * wavelength_m**2 * cross_section_m2
    denominator = (4.0 * np.pi)**3 * BOLTZMANN * T0 * noise_bandwidth_hz * noise_factor * losses * min_snr
    return float((numerator / denominator) ** 0.25)


def range_interval_m(cfg: dict[str, Any]) -> tuple[float, float]:
    """``(pessimistic, optimistic)`` detection ranges (m) of the configuration."""
    link_budget = cfg["link_budget"]
    f_c_hz = float(cfg["sdr"]["center_frequency_hz"])
    wavelength_m = SPEED_OF_LIGHT / f_c_hz
    noise_bandwidth_hz = float(link_budget["noise_bandwidth_hz"])

    range_max_m = radar_range_m(link_budget["optimistic"], wavelength_m, noise_bandwidth_hz)
    range_min_m = radar_range_m(link_budget["pessimistic"], wavelength_m, noise_bandwidth_hz)

    logger.info(
        "Detection range — %.1f m (pessimistic) / %.1f m (optimistic) "
        "(B=%.3f Hz, λ=%.3f m)",
        range_min_m,
        range_max_m,
        noise_bandwidth_hz,
        wavelength_m,
    )
    return range_min_m, range_max_m
