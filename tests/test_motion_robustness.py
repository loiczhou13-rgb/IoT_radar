"""Robustness to residual motion (MOTION_ROBUSTNESS_PLAN.md): M1 Doppler filter, M2 centre tracking,
M3 weighting, M4 measurement mode."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pytest

from characterization_cases import pipeline_config
from iot_radar.acquisition.pluto import effective_tx_offset_hz
from iot_radar.acquisition.sources import open_source
from iot_radar.dsp.decimation import Decimator
from iot_radar.dsp.doppler import alias_audit, composite_alias_attenuation, doppler_lowpass_decimator
from iot_radar.pipeline import MotionRobustnessSettings, VitalSignsPipeline

F_IN_HZ = 100e3          # reduced input rate of the tests (decimation 1000 down to 100 Hz)


def _m1_chain() -> tuple[Decimator, Decimator, float]:
    """Decimation 100 kS/s → 100 Hz (Chebyshev) then M1 → 20 Hz, as in the pipeline."""
    to_doppler = Decimator(F_IN_HZ, 1000, 5.0, output_dtype=np.complex128)
    m1, delay_s = doppler_lowpass_decimator(100.0, 5)
    return to_doppler, m1, delay_s


def _output_level_db(frequency_hz: float, duration_s: float = 12.0) -> float:
    """Level (dB) of a unit complex tone at the output of the M1 chain, in steady state."""
    to_doppler, m1, delay_s = _m1_chain()
    t = np.arange(int(duration_s * F_IN_HZ)) / F_IN_HZ
    tone = np.exp(2j * np.pi * frequency_hz * t)
    out = np.concatenate([m1(to_doppler(chunk)) for chunk in np.array_split(tone, 30)])
    steady = out[int(2 * delay_s * 20) + 20:]
    return float(20 * np.log10(np.sqrt(np.mean(np.abs(steady) ** 2)) + 1e-300))


# ---------------------------------------------------------------------------
# M1 — Doppler filter and aliasing
# ---------------------------------------------------------------------------

def test_m1_chain_meets_the_aliasing_requirement() -> None:
    to_doppler, m1, delay_s = _m1_chain()
    stages = to_doppler.stages + m1.stages
    assert min(s.worst_attenuation_db for s in alias_audit(stages, 1.0)) >= 60.0
    worst_db, _ = composite_alias_attenuation(stages, 1.0, max_frequency_hz=200.0)
    assert worst_db >= 60.0
    assert delay_s == pytest.approx(1.82)            # (365 − 1) / 2 samples at 100 Hz


@pytest.mark.parametrize("frequency_hz", [19.0, -19.0, 21.0, 39.0, -41.0, 59.0, 99.5, 3.0, 8.0, 16.0])
def test_tones_folding_into_the_band_and_gestures_are_attenuated_by_60_db(frequency_hz: float) -> None:
    assert _output_level_db(frequency_hz) <= -60.0


def test_the_former_chain_lets_gestures_through() -> None:
    """Without M1, an 8 Hz gesture reaches the slow time almost intact (audit of the plan, §1.2)."""
    decimator = Decimator(F_IN_HZ, 5000, 5.0)
    t = np.arange(int(10 * F_IN_HZ)) / F_IN_HZ
    out = np.concatenate([decimator(c) for c in np.array_split(np.exp(2j * np.pi * 8.0 * t), 20)])[60:]
    assert 20 * np.log10(np.sqrt(np.mean(np.abs(out) ** 2))) > -1.0


def test_breathing_components_pass_the_m1_chain() -> None:
    """The M1 filter is flat (0 dB) over the breathing band and its harmonics; the whole chain too.

    The Chebyshev stages before M1 have a constant gain of about −0.3 dB
    (0.05 dB of ripple each), as in the former chain: it does not change the
    phase, hence not the displacement.
    """
    from scipy.signal import freqz

    _, m1, _ = _m1_chain()
    taps = m1.stages[0][0]
    frequencies = np.array([0.1, 0.3, 0.5, 0.8, 1.0])
    _, response = freqz(taps, 1, worN=frequencies, fs=100.0)
    np.testing.assert_allclose(20 * np.log10(np.abs(response)), 0.0, atol=0.02)
    levels = [_output_level_db(f, duration_s=30.0) for f in (0.1, 0.3, -0.5, 1.0)]
    assert max(levels) - min(levels) < 0.05


def _config(motion: dict[str, Any], **simulation: Any) -> dict[str, Any]:
    cfg = pipeline_config()
    cfg["vital_signs"] = {}
    cfg["slow_time"] = {"rate_hz": 20.0, "warmup_s": 1.0}
    cfg["motion_robustness"] = motion
    cfg["simulation"].update(seed=3, snr_db=-10.0, breath_rate_hz=0.3, breath_amplitude_mm=4.0)
    cfg["simulation"].update(simulation)
    return cfg


M1_ONLY = {"doppler_filter": {"enabled": True}}


def _run(cfg: dict[str, Any], duration_s: float):
    logging.disable(logging.WARNING)
    source = open_source(cfg, simulation=True)
    pipeline = VitalSignsPipeline.from_config(cfg, source.sample_rate_hz, effective_tx_offset_hz(cfg))
    outputs = []
    for output in pipeline.run(source):
        outputs.append(output)
        if output.t_s >= duration_s:
            break
    logging.disable(logging.NOTSET)
    return outputs, pipeline


def test_breathing_is_recovered_through_m1() -> None:
    outputs, pipeline = _run(_config(M1_ONLY), 45.0)
    assert isinstance(pipeline.motion, MotionRobustnessSettings) and pipeline.motion.doppler_filter_enabled
    last = outputs[-1]
    assert last.breathing.state == "BREATHING"
    assert last.breathing.rate_bpm == pytest.approx(18.0, abs=1.0)
    peak_to_peak_from_std_mm = 2 * np.sqrt(2) * np.std(last.analysis.breath_mm)
    assert peak_to_peak_from_std_mm == pytest.approx(8.0, rel=0.10)


# ---------------------------------------------------------------------------
# M2 — sliding centre of the arc
# ---------------------------------------------------------------------------

from iot_radar.dsp.phase import demodulate, demodulate_tracked, fit_circle, fit_circle_pratt, track_center  # noqa: E402

LAMBDA_M = 299_792_458.0 / 2.45e9
F_SLOW_HZ = 20.0


def _arc(drift: complex, n_seconds: float = 20.0, snr_db: float = 30.0, seed: int = 0):
    """Slow-time IQ of 8 mm peak-to-peak breathing around a centre drifting by *drift* (radius 1)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_seconds * F_SLOW_HZ)) / F_SLOW_HZ
    displacement_m = 0.004 * np.sin(2 * np.pi * 0.25 * t)
    centre = 30.0 * np.exp(0.4j) + drift * t / t[-1]
    noise = 10 ** (-snr_db / 20) / np.sqrt(2) * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
    z = centre + np.exp(1j * (1.1 - 4 * np.pi * displacement_m / LAMBDA_M)) + noise
    return z, displacement_m


def _displacement_error_mm(phase_rad: np.ndarray, truth_m: np.ndarray) -> float:
    estimate = -phase_rad * LAMBDA_M / (4 * np.pi)
    error = (estimate - estimate.mean()) - (truth_m - truth_m.mean())
    return float(np.sqrt(np.mean(error ** 2)) * 1e3)


def test_pratt_fit_agrees_with_the_geometric_fit() -> None:
    z, _ = _arc(drift=0.0, snr_db=40.0)
    pratt, geometric = fit_circle_pratt(z), fit_circle(z)
    assert pratt.valid and geometric.valid
    assert abs(pratt.center - geometric.center) < 0.02 and pratt.radius == pytest.approx(geometric.radius, rel=0.02)


def test_drifting_centre_is_tracked_without_phase_jumps() -> None:
    """The centre moves by one radius during the window (a person changes the static reflections)."""
    z, truth_m = _arc(drift=1.0 * np.exp(2.0j))
    fixed = demodulate(z, "arctan", "circle", 0.3, 0.3)
    tracked = demodulate_tracked(z, F_SLOW_HZ, "arctan", window_s=10.0, step_s=1.0)
    assert tracked.method == "arctan_tracked"
    assert _displacement_error_mm(tracked.phase_rad, truth_m) < 1.0
    assert not fixed.calibrated or _displacement_error_mm(fixed.phase_rad, truth_m) > 2.0
    undetrended = track_center(z, F_SLOW_HZ, window_s=10.0, step_s=1.0, detrend=False)
    assert _displacement_error_mm(np.unwrap(np.angle(z - undetrended.center_per_sample)), truth_m) > 5.0
    # Continuity: no step between two samples beyond the breathing motion itself (≈ 0.04 rad here).
    assert np.max(np.abs(np.diff(tracked.phase_rad))) < 0.15
    track = track_center(z, F_SLOW_HZ, window_s=10.0, step_s=1.0)
    assert track.valid.all() and len(track.times_s) == 11
    assert np.max(np.abs(np.diff(track.center_per_sample))) < 0.02     # interpolated: no step


@pytest.mark.parametrize("method", ["pratt", "geometric", "kasa"])
def test_centre_estimators_can_be_selected(method: str) -> None:
    z, _ = _arc(drift=0.0)
    assert track_center(z, F_SLOW_HZ, method=method) is not None
    with pytest.raises(ValueError):
        track_center(z, F_SLOW_HZ, method="median")


def test_without_a_valid_sub_window_the_former_demodulation_is_used() -> None:
    rng = np.random.default_rng(1)
    noise = 30.0 + rng.standard_normal(400) + 1j * rng.standard_normal(400)
    result = demodulate_tracked(noise, F_SLOW_HZ)
    assert result.method == "linear" and not result.calibrated


def test_with_m1_an_empty_room_is_not_detected() -> None:
    """M1 empties the former reference band (2.5–5 Hz): the noise floor is measured in the pass band."""
    outputs, _ = _run(_config(M1_ONLY, presence=False, seed=1), 60.0)
    analysed = [o for o in outputs if o.analysis is not None]
    assert all(o.breathing.state != "BREATHING" for o in analysed)
    assert np.median([o.analysis.metrics.snr_db for o in analysed]) < 10.0
