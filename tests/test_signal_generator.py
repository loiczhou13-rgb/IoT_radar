"""The independent signal generator of the tests (tests/signal_generator.py) matches its specification."""

from __future__ import annotations

import numpy as np
import pytest

from signal_generator import CARRIER_HZ, SPEED_OF_LIGHT_M_S, RadarScene, blocks, breathing, with_apnea


def test_exact_path_length_and_quantisation() -> None:
    """Noise-free echo: the phase of the echo line follows −4π·f_c·R/c; samples are 12-bit integers."""
    scene = RadarScene(range_m=2.5, displacement_m=lambda t: np.zeros_like(t), amplitude_variation=0.0,
                       leakage=0.0, receiver_dc=0.0, echo_amplitude=1000.0, noise_density_db=300.0)
    z = scene.samples(0, 20_000).astype(np.complex128)
    t_s = np.arange(z.size) / scene.sample_rate_hz
    echo = np.mean(z * np.exp(-2j * np.pi * scene.tx_offset_hz * t_s))
    tau_s = 2 * 2.5 / SPEED_OF_LIGHT_M_S
    expected = np.angle(np.exp(-2j * np.pi * (CARRIER_HZ + scene.tx_offset_hz) * tau_s))
    assert np.angle(echo * np.conj(np.exp(1j * expected))) == pytest.approx(0.0, abs=2e-3)
    assert np.all(z.real == np.round(z.real)) and z.real.max() <= 2047 and z.real.min() >= -2048


def test_amplitude_variation_and_leakage() -> None:
    scene = RadarScene(seed=3)
    amplitude = scene.echo_amplitude_at(np.arange(0, 120, 0.1))
    assert 0.35 < amplitude.min() < 0.7 and 1.3 < amplitude.max() < 1.65      # up to ±60 %
    assert abs(scene.leakage) > 100 * scene.echo_amplitude


def test_breathing_rate_is_not_a_whole_number_of_cycles_per_window() -> None:
    assert (0.27 * 30.0) % 1.0 == pytest.approx(0.1)
    apnea = with_apnea(breathing(0.27, 8.0), 10.0, 20.0)
    t_s = np.linspace(10.0, 19.9, 50)
    assert np.ptp(apnea(t_s)) == 0.0


def test_losses_do_not_show_in_the_sample_counter() -> None:
    quiet = dict(seed=1, noise_density_db=300.0)
    out = list(blocks(RadarScene(**quiet), 1.0, block_size=2000, lost={4: 500}, flagged={2}))
    assert [start for _, start, _ in out] == [2000 * k for k in range(len(out))]
    assert [flag for _, _, flag in out] == [k == 2 for k in range(len(out))]
    np.testing.assert_array_equal(out[4][0][0, :10], RadarScene(**quiet).samples(8500, 10))   # 500 samples lost
