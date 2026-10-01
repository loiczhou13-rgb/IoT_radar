"""Breathing metrics and the temporal decision layer (iot_radar.dsp.detection)."""

from __future__ import annotations

import numpy as np
import pytest

from iot_radar.dsp.detection import (
    STATE_BREATHING,
    STATE_MOTION,
    STATE_NO_BREATHING,
    BreathingDetector,
    BreathMetrics,
    DetectionSettings,
    breathing_metrics,
)
from iot_radar.dsp.estimation import periodogram
from iot_radar.dsp.filters import detrend

F_S_HZ = 20.0
BANDS = dict(breathing_band_hz=(0.1, 0.5), reference_band_hz=(2.5, 5.0))


def _metrics(confidence: float, motion: bool = False, rate_hz: float = 0.25) -> BreathMetrics:
    return BreathMetrics(rate_hz, True, 20.0, 1.0, 0.9, motion, 8.0, 1.0, 1.0, confidence)


def test_metrics_of_breathing_and_of_noise() -> None:
    rng = np.random.default_rng(0)
    t_s = np.arange(400) / F_S_HZ
    breathing_mm = 4.0 * np.sin(2 * np.pi * 0.25 * t_s) + 0.05 * rng.standard_normal(t_s.size)
    spectrum = periodogram(detrend(breathing_mm), F_S_HZ, n_fft=4096)
    metrics = breathing_metrics(spectrum, 8.0, settings=DetectionSettings(), **BANDS)
    assert metrics.rate_hz == pytest.approx(0.25, abs=0.01)
    assert metrics.genuine_peak and not metrics.motion
    assert metrics.confidence > 0.9

    noise_mm = rng.standard_normal(t_s.size)
    noise = breathing_metrics(periodogram(noise_mm, F_S_HZ, n_fft=4096), 3.0,
                              settings=DetectionSettings(), **BANDS)
    assert noise.confidence < 0.6

    moving = breathing_metrics(spectrum, 80.0, settings=DetectionSettings(), **BANDS)
    assert moving.motion and moving.confidence == 0.0


def test_settings_from_config() -> None:
    settings = DetectionSettings.from_config({"threshold_on": 0.7, "unknown_key": 1})
    assert settings.threshold_on == 0.7 and settings.threshold_off == 0.4
    with pytest.raises(ValueError):
        DetectionSettings.from_config({"threshold_on": 0.3, "threshold_off": 0.5})


def test_hysteresis_and_smoothing() -> None:
    detector = BreathingDetector(DetectionSettings(), period_s=0.5)
    states = [detector.update(_metrics(1.0), 0.5 * k).state for k in range(20)]
    assert states[0] == STATE_NO_BREATHING  # no instantaneous detection
    assert states[-1] == STATE_BREATHING
    for k in range(20, 40):  # a dip between the OFF and ON thresholds keeps BREATHING
        state = detector.update(_metrics(0.5), 0.5 * k)
    assert state.state == STATE_BREATHING and state.rate_bpm == pytest.approx(15.0)
    for k in range(40, 80):
        state = detector.update(_metrics(0.0), 0.5 * k)
    assert state.state == STATE_NO_BREATHING and state.rate_bpm is None


def test_motion_state_and_apnea_alert() -> None:
    detector = BreathingDetector(DetectionSettings(), period_s=0.5)
    state = detector.update(_metrics(0.0, motion=True), 0.0)
    assert state.state == STATE_MOTION and not state.detected

    detector = BreathingDetector(DetectionSettings(), period_s=0.5)
    for k in range(20):
        detector.update(_metrics(1.0), 0.5 * k, since_last_breath_s=2.0)
    assert not detector.update(_metrics(1.0), 10.0, since_last_breath_s=3.0).apnea
    assert detector.update(_metrics(0.3), 10.5, since_last_breath_s=12.0).apnea
