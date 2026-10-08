"""Characterization cases: deterministic inputs fed to the radar DSP code.

Each ``case_*`` function runs one part of the processing chain on a fixed,
reproducible input and returns a dictionary of NumPy arrays.  The arrays
produced by the original code (commit 8189bbd) are stored in
``tests/data/references.npz`` by ``tests/make_references.py``;
``tests/test_characterization.py`` checks that the current code still
produces **bit-identical** values.

These references must never be regenerated, except in the commits that fix
bugs B2 and B5 of ``REFACTOR_PLAN.md`` (they change the simulation on
purpose).  The cases of the removed micro-Doppler chain (clutter filter,
Fisher x ACF detection, streaming pipeline, its dashboard context) were
dropped together with that chain.

The ``cw_simulation`` and ``vital_signs_pipeline`` cases were added before
the motion-robustness work (``MOTION_ROBUSTNESS_PLAN.md``): with M1 to M3
disabled, the simulation without disturbers and the phase pipeline must
stay unchanged bit for bit.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from iot_radar.acquisition.pluto import cw_tx_buffer
from iot_radar.acquisition.sources import Block, CWSimulationSource
from iot_radar.dsp.decimation import Decimator
from iot_radar.dsp.detection import STATE_CODES
from iot_radar.dsp.spectral import compute_single_column, get_window
from iot_radar.physics import range_interval_m
from iot_radar.pipeline import VitalSignsPipeline

# Reduced configuration (fast) of the radar, used by several tests.  The TX
# offset 244.140625 Hz is exactly 10 periods of a 4096-sample buffer at
# 100 kHz, so the cyclic TX buffer is continuous (see bug B1 in
# REFACTOR_PLAN.md).
PIPELINE_CONFIG: dict[str, Any] = {
    "logging": {"level": "WARNING", "to_file": False},
    "sdr": {
        "uri": "ip:192.168.2.1",
        "center_frequency_hz": 3.5e9,
        "sample_rate_hz": 100e3,
        "rx_gain_db": 45,
        "tx_gain_db": -20,
        "buffer_size": 4096,
    },
    "tx": {"waveform": "cw_offset", "offset_hz": 244.140625},
    "link_budget": {
        "optimistic": {
            "tx_power_dbm": -13, "tx_antenna_gain_dbi": 2, "rx_antenna_gain_dbi": 2,
            "radar_cross_section_m2": 0.5, "noise_figure_db": 4, "system_losses_db": 3,
            "min_snr_db": 3,
        },
        "pessimistic": {
            "tx_power_dbm": -13, "tx_antenna_gain_dbi": 0, "rx_antenna_gain_dbi": 0,
            "radar_cross_section_m2": 0.05, "noise_figure_db": 6, "system_losses_db": 25,
            "min_snr_db": 3,
        },
        "noise_bandwidth_hz": 50,
    },
    "recording": {"sessions_dir": "data/sessions", "flush_interval_s": 1.0},
    "simulation": {
        "enabled": True,
        "presence": True,
        "breath_rate_hz": 0.3,
        "breath_amplitude_mm": 10,
        "target_range_m": 3.0,
        "target_amplitude": 1.0,
        "static_clutter_amplitude": 30.0,
        "receiver_dc_amplitude": 50.0,
        "lo_offset_hz": 0.0,
        "snr_db": -25.0,
        "realtime": False,
    },
}

def pipeline_config() -> dict[str, Any]:
    """Return a fresh copy of the reduced pipeline configuration."""
    return copy.deepcopy(PIPELINE_CONFIG)


def _complex_noise(n: int, seed: int) -> np.ndarray:
    """Deterministic complex white Gaussian noise (complex64)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def _test_signal(n: int, f_s_hz: float, seed: int) -> np.ndarray:
    """DC offset + slow phase-modulated tone + noise (complex64)."""
    t = np.arange(n) / f_s_hz
    tone = np.exp(1j * (2 * np.pi * 37.0 * t - 1.2 * np.sin(2 * np.pi * 0.3 * t)))
    return (100.0 + tone + 0.1 * _complex_noise(n, seed)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_decimator() -> dict[str, np.ndarray]:
    """Stateful decimation, fed in uneven blocks, for two factors."""
    x = _test_signal(300_000, 2e6, seed=1)
    out = {}
    for factor in (1000, 100):
        decimator = Decimator(f_s_hz=2e6, decimation_factor=factor, max_frequency_hz=10.0)
        blocks = np.array_split(x, 13)
        out[f"d{factor}"] = np.concatenate([decimator(b) for b in blocks])
    return out


def case_windows() -> dict[str, np.ndarray]:
    """Every supported analysis window."""
    return {mode: get_window(mode, 64) for mode in ("hann", "hamming", "blackman", "flattop", "none")}


def case_spectral_column() -> dict[str, np.ndarray]:
    """One STFT column of a windowed segment."""
    segment = _test_signal(1024, 2000.0, seed=3)
    column = compute_single_column(segment, 2000.0, get_window("hann", 1024))
    return {"col_db": column.power_db, "f_hz": column.f_hz}


def case_tx_buffer() -> dict[str, np.ndarray]:
    """CW and CW-offset transmit buffers (offset = whole number of periods)."""
    return {
        "cw": cw_tx_buffer("cw", 1024, 2e6),
        "cw_offset": cw_tx_buffer("cw_offset", 16384, 2e6, 488.28125),
    }


def case_link_budget() -> dict[str, np.ndarray]:
    """Pessimistic and optimistic ranges of the radar equation."""
    return {"range_m": np.array(range_interval_m(pipeline_config()))}


def _reduced_simulation(seed: int, timeline: list | None = None) -> CWSimulationSource:
    """Simulated source of the reduced configuration (100 kS/s, 4096-sample blocks)."""
    sdr = PIPELINE_CONFIG["sdr"]
    return CWSimulationSource(
        f_c_hz=sdr["center_frequency_hz"], f_s_hz=sdr["sample_rate_hz"], buffer_size=sdr["buffer_size"],
        tx_offset_hz=PIPELINE_CONFIG["tx"]["offset_hz"], breath_rate_hz=0.3, breath_amplitude_mm=4.0,
        heart_amplitude_mm=0.2, timeline=timeline, snr_db=-12.0, lo_offset_hz=0.01, seed=seed,
    )


def case_cw_simulation() -> dict[str, np.ndarray]:
    """Blocks, ground truth and annotations of the single-channel simulation."""
    source = _reduced_simulation(seed=7, timeline=[[0.0, "breathing"], [0.05, "motion"], [0.08, "apnea"]])
    blocks = [source.read_block() for _ in range(3)]
    t_s = np.linspace(0.0, 0.12, 25)
    annotations = source.scene_annotations(3 * 4096)
    return {
        "samples": np.concatenate([b.samples for b in blocks], axis=1),
        "sample_start": np.array([b.sample_start for b in blocks]),
        "chest_displacement_m": source.chest_displacement_m(t_s),
        "annotation_bounds": np.array([(start, count) for start, count, _ in annotations]),
    }


def case_vital_signs_pipeline() -> dict[str, np.ndarray]:
    """45 s of the single-channel phase pipeline, with one stream discontinuity."""
    source = _reduced_simulation(seed=11)
    pipeline = VitalSignsPipeline.from_config(
        pipeline_config(), PIPELINE_CONFIG["sdr"]["sample_rate_hz"], PIPELINE_CONFIG["tx"]["offset_hz"],
    )
    outputs = []
    n_blocks = int(45.0 * 100e3 / 4096)
    for index in range(n_blocks):
        block = source.read_block()
        if index == n_blocks // 3:  # lost samples before this block
            block = Block(block.samples, block.sample_start, block.host_time_s, True)
        outputs.extend(pipeline.process_block(block))
    last = outputs[-1].analysis
    rates = [o.breathing.rate_bpm for o in outputs]
    return {
        "t_s": np.array([o.t_s for o in outputs]),
        "discontinuity": np.array([o.discontinuity for o in outputs]),
        "state": np.array([STATE_CODES[o.breathing.state] for o in outputs]),
        "confidence": np.array([o.breathing.confidence for o in outputs]),
        "rate_bpm": np.array([np.nan if r is None else r for r in rates]),
        "last_displacement_mm": last.displacement_mm,
        "last_breath_mm": last.breath_mm,
        "last_iq": last.iq,
        "last_spectrum_psd": last.spectrum_psd,
        "last_metric_rate_hz": np.array(last.metrics.rate_hz),
        "last_micro_doppler_db": outputs[-1].micro_doppler_column_db,
    }


CASES: dict[str, Callable[[], dict[str, np.ndarray]]] = {
    "decimator": case_decimator,
    "windows": case_windows,
    "spectral_column": case_spectral_column,
    "tx_buffer": case_tx_buffer,
    "link_budget": case_link_budget,
    "cw_simulation": case_cw_simulation,
    "vital_signs_pipeline": case_vital_signs_pipeline,
}
