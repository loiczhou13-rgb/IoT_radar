"""STFT columns of the micro-Doppler view (iot_radar.dsp.spectral)."""

from __future__ import annotations

import numpy as np

from iot_radar.dsp.spectral import compute_single_column, get_window


def test_zero_padded_column() -> None:
    column = compute_single_column(np.ones(64, np.complex64), 20.0, get_window("hann", 64), n_fft=256)
    assert column.power_db.shape == column.f_hz.shape == (256,)
    assert column.f_hz[np.argmax(column.power_db)] == 0.0
