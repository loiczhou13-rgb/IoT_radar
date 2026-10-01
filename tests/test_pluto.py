"""PlutoSDR helpers: TX waveform (bug B1) and effective TX offset."""

from __future__ import annotations

import numpy as np
import pytest

from iot_radar.acquisition.pluto import generate_tx_buffer, resolve_f_offset, snap_tx_offset
from iot_radar.config import DEFAULT_RADAR_CONFIG, load_config


def test_snap_tx_offset_gives_whole_periods() -> None:
    f_eff = snap_tx_offset(500.0, 2e6, 16384)
    assert f_eff == pytest.approx(488.28125)
    assert f_eff * 16384 / 2e6 == pytest.approx(4.0)
    with pytest.raises(ValueError):
        snap_tx_offset(50.0, 2e6, 16384)  # below half a grid step


def test_cyclic_tx_buffer_is_continuous() -> None:
    f_eff = snap_tx_offset(500.0, 2e6, 16384)
    tx = generate_tx_buffer("cw_offset", 16384, 2e6, f_eff)
    next_sample = tx[-1] * np.exp(2j * np.pi * f_eff / 2e6)  # sample after the last one
    assert abs(next_sample - tx[0]) / abs(tx[0]) < 1e-4


def test_tx_buffer_refuses_a_fractional_number_of_periods() -> None:
    with pytest.raises(ValueError):
        generate_tx_buffer("cw_offset", 16384, 2e6, 500.0)


def test_default_config_transmits_and_receives_the_snapped_offset() -> None:
    cfg = load_config(DEFAULT_RADAR_CONFIG)
    assert resolve_f_offset(cfg) == pytest.approx(488.28125)
    cfg["emission"]["mode"] = "cw"
    assert resolve_f_offset(cfg) == 0.0
