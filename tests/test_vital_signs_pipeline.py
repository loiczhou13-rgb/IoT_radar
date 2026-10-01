"""Validation of the phase-demodulation pipeline on the simulated radar.

A reduced configuration keeps the tests fast: 100 kS/s instead of 2 MS/s
(decimation 5000 instead of 100000 down to the 20 Hz slow time), with the
default processing and detection settings of configs/radar.yaml.  The noise
(-10 dB per ADC sample) gives a slow-time SNR of about 27 dB.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pytest

from iot_radar.acquisition.pluto import effective_tx_offset_hz
from iot_radar.acquisition.sources import Block, CWSimulationSource, open_source
from iot_radar.config import DEFAULT_RADAR_CONFIG, load_config
from iot_radar.dsp.detection import STATE_BREATHING, STATE_MOTION, STATE_NO_BREATHING, STATE_WARMUP
from iot_radar.pipeline import PipelineOutput, VitalSignsPipeline

BREATH_RATE_HZ = 0.3          # 18 breaths/min
BREATH_AMPLITUDE_MM = 4.0     # 8 mm peak to peak


def _config(**simulation: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(DEFAULT_RADAR_CONFIG))
    cfg["sdr"].update(sample_rate_hz=100e3, buffer_size=4096)
    cfg["tx"]["offset_hz"] = 244.140625  # 10 periods per 4096-sample buffer
    cfg["simulation"].update(
        enabled=True, realtime=False, seed=3, snr_db=-10.0,
        breath_rate_hz=BREATH_RATE_HZ, breath_amplitude_mm=BREATH_AMPLITUDE_MM,
    )
    cfg["simulation"].update(simulation)
    return cfg


def _run(cfg: dict[str, Any], duration_s: float, source: Any = None) -> list[PipelineOutput]:
    """Outputs of the pipeline on *duration_s* seconds of the simulated stream."""
    logging.disable(logging.WARNING)
    source = source or open_source(cfg, simulation=True)
    pipeline = VitalSignsPipeline.from_config(cfg, source.sample_rate_hz, effective_tx_offset_hz(cfg))
    outputs = []
    for output in pipeline.run(source):
        outputs.append(output)
        if output.t_s >= duration_s:
            break
    logging.disable(logging.NOTSET)
    return outputs


def _at(outputs: list[PipelineOutput], t_s: float) -> PipelineOutput:
    return min(outputs, key=lambda o: abs(o.t_s - t_s))


def test_breathing_rate_within_one_breath_per_minute() -> None:
    outputs = _run(_config(), 40.0)
    last = outputs[-1]
    assert last.breathing.state == STATE_BREATHING
    assert last.breathing.rate_bpm == pytest.approx(60 * BREATH_RATE_HZ, abs=1.0)
    assert last.analysis.demodulation.method == "arctan" and last.analysis.demodulation.calibrated


def test_displacement_amplitude_is_recovered() -> None:
    analysis = _run(_config(), 40.0)[-1].analysis
    # Peak-to-peak amplitude of a sinusoid = 2·sqrt(2)·std (the noise adds a little).
    peak_to_peak_from_std_mm = 2 * np.sqrt(2) * np.std(analysis.breath_mm)
    assert peak_to_peak_from_std_mm == pytest.approx(2 * BREATH_AMPLITUDE_MM, rel=0.10)
    assert analysis.cycles.depth == pytest.approx(2 * BREATH_AMPLITUDE_MM, rel=0.10)


def test_empty_scene_is_not_detected() -> None:
    outputs = _run(_config(presence=False), 40.0)
    assert all(o.breathing.state in (STATE_WARMUP, STATE_NO_BREATHING) for o in outputs)


def test_apnea_raises_the_alert() -> None:
    outputs = _run(_config(timeline=[[0, "breathing"], [35, "apnea"]], heart_amplitude_mm=0.15), 50.0)
    assert not _at(outputs, 34.0).breathing.apnea
    assert _at(outputs, 34.0).breathing.state == STATE_BREATHING
    # The last inhalation before the apnea ends at 32.5 s (chest maximum of
    # the 0.3 Hz breathing); apnea_s = 10 s, so the alert comes at ~42.5 s.
    alert_times_s = [o.t_s for o in outputs if o.breathing.apnea]
    assert alert_times_s and 42.0 <= alert_times_s[0] <= 44.0
    assert all(o.breathing.apnea for o in outputs if o.t_s >= 44.0)


def test_motion_is_flagged() -> None:
    outputs = _run(_config(timeline=[[0, "breathing"], [35, "motion"]]), 41.0)
    assert _at(outputs, 34.0).breathing.state == STATE_BREATHING
    assert _at(outputs, 40.0).breathing.state == STATE_MOTION


class _DroppingSource:
    """Simulated source that loses samples once (like an RX overflow)."""

    def __init__(self, source: CWSimulationSource, drop_after_s: float) -> None:
        self._source = source
        self.sample_rate_hz = source.sample_rate_hz
        self._drop_at = int(drop_after_s * source.sample_rate_hz)
        self._delivered = 0
        self._dropped = False

    def read_block(self) -> Block:
        block = self._source.read_block()
        if not self._dropped and block.sample_start >= self._drop_at:
            self._dropped = True
            for _ in range(5):  # 0.2 s of signal never reaches the host
                block = self._source.read_block()
            block = Block(block.samples, self._delivered, block.host_time_s, True)
        else:
            block = Block(block.samples, self._delivered, block.host_time_s, False)
        self._delivered += block.samples.shape[1]
        return block

    def close(self) -> None:
        self._source.close()


def test_stream_discontinuity_restarts_the_processing() -> None:
    cfg = _config()
    source = _DroppingSource(open_source(cfg, simulation=True), drop_after_s=30.0)
    outputs = _run(cfg, 65.0, source=source)
    restarts = [o for o in outputs if o.discontinuity]
    assert len(restarts) == 1
    t_restart = restarts[0].t_s
    assert _at(outputs, t_restart - 1.0).breathing.state == STATE_BREATHING
    assert restarts[0].breathing.state == STATE_WARMUP  # the window refills from scratch
    assert _at(outputs, t_restart + 15.0).breathing.state == STATE_WARMUP
    assert _at(outputs, t_restart + 30.0).breathing.state == STATE_BREATHING


def test_slow_input_rate_of_converted_recordings() -> None:
    """2 kHz input (converted legacy recordings): decimation by 100."""
    cfg = _config()
    pipeline = VitalSignsPipeline.from_config(cfg, input_rate_hz=2000.0, tx_offset_hz=0.0)
    t_s = np.arange(int(30 * 2000)) / 2000.0
    displacement_m = 4e-3 * np.sin(2 * np.pi * 0.25 * t_s)
    wavelength_m = 299_792_458.0 / 3.5e9
    iq = (40.0 + np.exp(-4j * np.pi * displacement_m / wavelength_m)).astype(np.complex64)
    logging.disable(logging.WARNING)
    outputs = []
    for start in range(0, iq.size, 4000):
        block = Block(iq[np.newaxis, start:start + 4000], start, 0.0, False)
        outputs += pipeline.process_block(block)
    logging.disable(logging.NOTSET)
    assert outputs[-1].analysis.metrics.rate_hz == pytest.approx(0.25, abs=0.01)
    assert outputs[-1].micro_doppler_column_db.shape == (256,)
