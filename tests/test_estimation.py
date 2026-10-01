"""Breathing-rate estimators and breath-by-breath analysis (iot_radar.dsp.estimation)."""

from __future__ import annotations

import numpy as np
import pytest

from iot_radar.dsp import estimation

F_S_HZ = 20.0
BAND_HZ = (0.1, 0.5)


@pytest.mark.parametrize("rate_hz", [0.15, 0.25, 0.4])
def test_all_estimators_on_a_sinusoid(rate_hz: float) -> None:
    t_s = np.arange(400) / F_S_HZ
    x = np.sin(2 * np.pi * rate_hz * t_s)
    assert estimation.fft_peak_rate(x, F_S_HZ, BAND_HZ)[0] == pytest.approx(rate_hz, abs=0.01)
    assert estimation.acf_rate(x, F_S_HZ, BAND_HZ)[0] == pytest.approx(rate_hz, abs=0.01)
    assert estimation.peak_count_rate(x, F_S_HZ, BAND_HZ) == pytest.approx(rate_hz, abs=0.02)
    assert estimation.zero_crossing_rate(x, F_S_HZ, BAND_HZ) == pytest.approx(rate_hz, abs=0.02)


def test_cycles_of_an_asymmetric_breath() -> None:
    # Chest expansion (−d) with a 1:2 inhalation/exhalation ratio, 0.25 Hz, 8 mm peak to peak.
    t_s = np.arange(400) / F_S_HZ
    cycle_phase = np.mod(t_s * 0.25, 1.0)
    chest_mm = np.where(cycle_phase < 1 / 3, cycle_phase * 3, (1 - cycle_phase) * 1.5) * 8.0
    cycles = estimation.breath_cycles(-(chest_mm - chest_mm.mean()), F_S_HZ, BAND_HZ)
    assert cycles.rate_hz == pytest.approx(0.25, abs=0.005)
    assert cycles.ie_ratio == pytest.approx(0.5, abs=0.05)
    assert cycles.depth == pytest.approx(8.0, abs=0.3)
    assert cycles.variability < 0.02
    assert cycles.since_last_s < 4.0


def test_periodogram_axis_and_padding() -> None:
    spectrum = estimation.periodogram(np.ones(100), F_S_HZ, n_fft=256)
    assert spectrum.f_hz.size == 129 and spectrum.f_hz[-1] == pytest.approx(10.0)
    assert spectrum.native_resolution_hz == pytest.approx(0.2)
    assert np.allclose(spectrum.psd, 0.0)  # the mean is removed
