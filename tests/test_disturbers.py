"""Simulated disturbers around the SISO radar (iot_radar.acquisition.disturbers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iot_radar.acquisition.disturbers import (
    Gesture,
    Sway,
    Walker,
    activity_intervals,
    disturber_echo,
    disturbers_from_config,
    path_masking,
)
from iot_radar.acquisition.recording import SessionReader, SessionWriter
from iot_radar.acquisition.sources import CWSimulationSource

LAMBDA_M = 299_792_458.0 / 2.45e9
F_S_HZ = 1000.0


def _dominant_frequency_hz(x: np.ndarray, f_s_hz: float) -> float:
    spectrum = np.abs(np.fft.fft(x * np.hanning(x.size)))
    return float(np.fft.fftfreq(x.size, 1 / f_s_hz)[np.argmax(spectrum)])


def test_walker_doppler_is_2v_over_lambda() -> None:
    """Receding at 1 m/s: the torso line is at −2v/λ ≈ −16.3 Hz (≈ 16·v at 2.45 GHz)."""
    walker = Walker(start_m=[0.0, 3.0], velocity_m_s=[0.0, 1.0])
    t_s = np.arange(0, 4.0, 1 / F_S_HZ)
    echo = disturber_echo(walker, t_s, LAMBDA_M) * (1.0 + 0.0 * t_s)
    # Compensate the 1/R² amplitude decay before the spectrum.
    range_m = np.linalg.norm(walker.torso_m(t_s), axis=1)
    assert _dominant_frequency_hz(echo * range_m ** 2, F_S_HZ) == pytest.approx(-2 * 1.0 / LAMBDA_M, abs=0.5)
    assert walker.gait_frequency_hz == pytest.approx(1.0 / 1.4)


def test_echo_amplitude_decays_as_one_over_range_squared() -> None:
    near, far = Sway([0.0, 2.0], amplitude_m=0.0), Sway([0.0, 4.0], amplitude_m=0.0)
    t_s = np.zeros(1)
    ratio = abs(disturber_echo(near, t_s, LAMBDA_M)[0]) / abs(disturber_echo(far, t_s, LAMBDA_M)[0])
    assert ratio == pytest.approx(4.0)


def test_gesture_doppler_reaches_16_hz() -> None:
    gesture = Gesture([0.0, 1.5], amplitude_m=0.08, frequency_hz=2.0)
    hand = gesture.scatterers(np.arange(0, 1.0, 1 / F_S_HZ)).positions_m[1]
    speed_m_s = np.abs(np.diff(np.linalg.norm(hand, axis=1))) * F_S_HZ
    assert 2 * speed_m_s.max() / LAMBDA_M == pytest.approx(16.4, abs=0.5)


def test_a_walker_crossing_the_line_of_sight_masks_the_target() -> None:
    walker = Walker(start_m=[-2.0, 1.5], velocity_m_s=[1.0, 0.0])     # crosses x = 0 at t = 2 s
    t_s = np.array([0.0, 2.0, 4.0])
    factor = path_masking(np.array([0.0, 3.0]), [walker], t_s)
    assert factor[0] == pytest.approx(1.0) and factor[2] == pytest.approx(1.0)
    assert 20 * np.log10(factor[1]) == pytest.approx(-20.0)
    behind = Walker(start_m=[-2.0, 5.0], velocity_m_s=[1.0, 0.0])      # behind the target: no masking
    assert path_masking(np.array([0.0, 3.0]), [behind], t_s)[1] == pytest.approx(1.0)


def test_activity_intervals() -> None:
    walker = Walker(start_m=[-6.0, 3.0], velocity_m_s=[1.0, 0.0])      # enters ±60° at x = −5.2 m
    sway = Sway([1.0, 2.0], start_s=10.0, stop_s=20.0)
    intervals = activity_intervals([walker, sway], n_samples=int(30 * F_S_HZ), sample_rate_hz=F_S_HZ)
    walk = [i for i in intervals if i[2] == "walker"][0]
    assert walk[0] / F_S_HZ == pytest.approx(6.0 - 3.0 * np.tan(np.radians(60)), abs=0.15)
    assert (walk[0] + walk[1]) / F_S_HZ == pytest.approx(6.0 + 3.0 * np.tan(np.radians(60)), abs=0.15)
    assert (10000, 10000, "sway") in intervals


def test_configuration_entries() -> None:
    disturbers = disturbers_from_config([
        {"kind": "walker", "start_m": [-4, 3], "velocity_m_s": [1.2, 0]},
        {"kind": "gesture", "position_m": [0.5, 1.5], "start_s": 5, "stop_s": 15},
    ])
    assert [d.kind for d in disturbers] == ["walker", "gesture"]
    with pytest.raises(ValueError):
        disturbers_from_config([{"kind": "dancer"}])


def _source(**kwargs) -> CWSimulationSource:
    return CWSimulationSource(f_c_hz=2.45e9, f_s_hz=20e3, buffer_size=4000, tx_offset_hz=500.0,
                              breath_rate_hz=0.3, breath_amplitude_mm=5.0, snr_db=-10.0, seed=3, **kwargs)


def test_simulation_with_disturbers_and_its_ground_truth(tmp_path: Path) -> None:
    walker = Walker(start_m=[-3.0, 1.5], velocity_m_s=[1.0, 0.0])
    plain, disturbed = _source(), _source(disturbers=[walker], static_scatterers=[{"position_m": [1, 5], "amplitude": 20}])
    a, b = plain.read_block(), disturbed.read_block()
    assert a.samples.shape == b.samples.shape and not np.array_equal(a.samples, b.samples)
    intervals = disturbed.disturbance_intervals(int(10 * 20e3))
    assert intervals and intervals[0][2] == "walker"

    radar = {"sample_rate_hz": 20e3, "center_frequency_hz": 2.45e9, "tx_waveform": "cw_offset", "tx_offset_hz": 500.0,
             "tx_gain_db": -20.0, "rx_gain_db": 45.0, "rx_gain_mode": "manual", "n_rx_channels": 1,
             "channel_layout": "rx0", "firmware_version": "simulation", "source_kind": "simulation",
             "iq_stage": "raw_adc"}
    with SessionWriter(tmp_path / "s.h5", radar, with_ground_truth=True) as writer:
        writer.write_block(b.samples, b.host_time_s, b.overflow)
        writer.set_disturbances(intervals)
    with SessionReader(tmp_path / "s.h5") as reader:
        assert reader.disturbances() == intervals
