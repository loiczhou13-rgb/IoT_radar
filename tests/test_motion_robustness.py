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
