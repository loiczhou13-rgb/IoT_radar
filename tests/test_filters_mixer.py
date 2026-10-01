"""Window filters and the NCO mixer."""

from __future__ import annotations

import numpy as np
import pytest

from iot_radar.dsp.filters import bandpass, detrend, widened_band
from iot_radar.dsp.mixer import Mixer


def test_bandpass_keeps_breathing_and_removes_drift() -> None:
    t_s = np.arange(400) / 20.0
    breathing = np.sin(2 * np.pi * 0.3 * t_s)
    disturbed = breathing + 0.5 * t_s + 0.1 * np.sin(2 * np.pi * 4.0 * t_s)
    filtered = bandpass(disturbed, 20.0, widened_band((0.1, 0.5), 20.0))
    assert np.std(filtered[50:-50] - breathing[50:-50]) < 0.05
    # Note: the odd extension over the whole window makes a strong
    # out-of-band component leak near the edges (its end values create a
    # low-frequency step); breathing dominates the real displacement.
    with pytest.raises(ValueError):
        bandpass(breathing, 20.0, (0.5, 0.1))


def test_detrend_removes_a_line() -> None:
    assert np.allclose(detrend(3.0 + 2.0 * np.arange(10)), 0.0)


def test_mixer_brings_the_tone_to_zero_whatever_the_blocks() -> None:
    f_s_hz, tone_hz = 1000.0, 123.0
    x = 5.0 * np.exp(2j * np.pi * tone_hz * np.arange(3000) / f_s_hz)
    whole = Mixer(f_s_hz, tone_hz)(x)
    mixer = Mixer(f_s_hz, tone_hz)
    pieces = np.concatenate([mixer(block) for block in np.array_split(x, 7)])
    np.testing.assert_allclose(whole, 5.0, atol=1e-9)
    np.testing.assert_allclose(pieces, whole, atol=1e-9)
    assert np.array_equal(Mixer(f_s_hz, 0.0)(x), x)
