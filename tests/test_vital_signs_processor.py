"""Window analysis of the phase chain (VitalSignsProcessor), on synthetic slow-time IQ."""

from __future__ import annotations

import numpy as np

from iot_radar.pipeline import VitalSignsProcessor, VitalSignsSettings
from test_phase import F_S_HZ, WAVELENGTH_M, breathing_iq


def _processor() -> VitalSignsProcessor:
    return VitalSignsProcessor(F_S_HZ, WAVELENGTH_M, VitalSignsSettings())


def test_breathing_gives_a_high_confidence() -> None:
    z, _ = breathing_iq(rate_hz=0.25)
    metrics = _processor().analyse(z).metrics
    assert metrics.confidence > 0.9
    assert abs(metrics.rate_hz - 0.25) < 0.01
    assert not metrics.motion


def test_noise_only_gives_a_low_confidence() -> None:
    confidences = [_processor().analyse(breathing_iq(echo=0.0, seed=seed)[0]).metrics.confidence
                   for seed in range(30)]
    assert max(confidences) < 0.6


def test_large_motion_is_flagged() -> None:
    t_s = np.arange(400) / F_S_HZ
    displacement_m = 0.05 * np.sin(2 * np.pi * 0.07 * t_s)  # 10 cm body movement
    z = 30 * np.exp(1.1j) + np.exp(-4j * np.pi * (2.0 + displacement_m) / WAVELENGTH_M)
    assert _processor().analyse(z).metrics.motion
