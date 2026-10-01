"""Phase demodulation bricks (iot_radar.dsp.phase), on synthetic slow-time IQ."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from iot_radar.dsp.phase import (
    dacm_demodulation,
    demodulate,
    fit_circle,
    phase_to_displacement_m,
    remove_common_rotation,
)
from iot_radar.physics import SPEED_OF_LIGHT

F_S_HZ = 20.0
WAVELENGTH_M = SPEED_OF_LIGHT / 3.5e9


def breathing_iq(
    rate_hz: float = 0.3,
    amplitude_m: float = 0.004,
    n: int = 400,
    clutter: complex = 30 * np.exp(1.1j),
    echo: float = 1.0,
    noise: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow-time IQ of a breathing target on top of a static DC offset, and its displacement."""
    rng = np.random.default_rng(seed)
    t_s = np.arange(n) / F_S_HZ
    displacement_m = amplitude_m * np.sin(2 * np.pi * rate_hz * t_s)
    z = clutter + echo * np.exp(-4j * np.pi * (2.0 + displacement_m) / WAVELENGTH_M)
    z = z + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return z, displacement_m


def test_circle_fit_recovers_dc_offset() -> None:
    z, _ = breathing_iq(noise=0.0)
    fit = fit_circle(z)
    assert fit.valid
    assert abs(fit.center - 30 * np.exp(1.1j)) < 1e-6
    assert abs(fit.radius - 1.0) < 1e-6


def test_arctan_after_circle_fit_gives_displacement() -> None:
    z, displacement_m = breathing_iq(noise=0.0)
    result = demodulate(z, "arctan", "circle")
    measured_m = phase_to_displacement_m(result.phase_rad, WAVELENGTH_M)
    error_m = (measured_m - measured_m.mean()) - (displacement_m - displacement_m.mean())
    assert np.max(np.abs(error_m)) < 1e-9


def test_dacm_matches_arctan() -> None:
    z, _ = breathing_iq(noise=0.0)
    centred = z - fit_circle(z).center
    arctan_rad = np.unwrap(np.angle(centred))
    dacm_rad = dacm_demodulation(centred)
    assert np.max(np.abs((arctan_rad - arctan_rad[0]) - dacm_rad)) < 0.05


def test_highpass_before_arctan_destroys_phase() -> None:
    """Bug B3 of the former chain: a clutter high-pass before angle() ruins the phase."""
    z, displacement_m = breathing_iq(noise=0.0, n=4000)
    sos = butter(2, 0.05, btype="high", fs=F_S_HZ, output="sos")
    filtered = sosfilt(sos, z.real) + 1j * sosfilt(sos, z.imag)
    after_highpass_m = phase_to_displacement_m(np.unwrap(np.angle(filtered[2000:])), WAVELENGTH_M)
    after_circle_fit_m = phase_to_displacement_m(demodulate(z[2000:]).phase_rad, WAVELENGTH_M)
    reference_m = displacement_m[2000:] - displacement_m[2000:].mean()
    error_highpass_m = np.std(after_highpass_m - after_highpass_m.mean() - reference_m)
    error_circle_m = np.std(after_circle_fit_m - after_circle_fit_m.mean() - reference_m)
    assert error_circle_m < 1e-6
    assert error_highpass_m > 1e-3  # millimetre-level garbage


def test_common_rotation_removed() -> None:
    z, _ = breathing_iq(noise=0.0)
    t_s = np.arange(z.size) / F_S_HZ
    _, drift_hz = remove_common_rotation(z * np.exp(2j * np.pi * 0.03 * t_s), F_S_HZ)
    assert abs(drift_hz - 0.03) < 1e-3


def test_linear_fallback_on_short_arc() -> None:
    z, _ = breathing_iq(amplitude_m=0.0002, noise=0.0)  # 0.03 rad arc
    assert demodulate(z).method == "linear"
