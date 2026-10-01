"""Characterization cases: deterministic inputs fed to the radar DSP code.

Each ``case_*`` function runs one part of the processing chain on a fixed,
reproducible input and returns a dictionary of NumPy arrays.  The arrays
produced by the original code (commit 8189bbd) are stored in
``tests/data/references.npz`` by ``tests/make_references.py``;
``tests/test_characterization.py`` checks that the current code still
produces **bit-identical** values.

These references must never be regenerated, except in the commits that fix
bugs B2 and B5 of ``REFACTOR_PLAN.md`` (they change the simulation on
purpose).
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from iot_radar.dsp.clutter import ClutterFilter
from iot_radar.dsp.decimation import Decimator
from iot_radar.dsp.detection import (
    _acf_peak,
    _fisher_p_value,
    _fusion_score,
    detect_presence_column,
)
from iot_radar.dsp.spectral import compute_single_column, get_window
from iot_radar.acquisition.pluto import cw_tx_buffer
from iot_radar.physics import range_interval_m
from iot_radar.acquisition.sources import open_source
from iot_radar.pipeline import build_context, streaming_frame_generator

SIMULATION_SEED = 1234
"""Seed of the random generator used by the simulated IQ source."""

# Reduced configuration (fast) of the streaming pipeline.  The TX offset
# 244.140625 Hz is exactly 10 periods of a 4096-sample buffer at 100 kHz, so
# the cyclic TX buffer is continuous (see bug B1 in REFACTOR_PLAN.md).
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
    "decimation": {"enabled": True, "factor": 100, "max_useful_frequency_hz": 10},
    "clutter": {
        "mode": "butterworth",
        "alpha": 0.9999,
        "butterworth_order": 2,
        "butterworth_cutoff_hz": 0.05,
    },
    "spectrogram": {"window": "hann", "n_fft": 4096, "overlap": 0.90, "skip_warmup_frames": 2},
    "detection": {
        "breathing_band_hz": [0.1, 0.8],
        "reference_band_hz": [2.0, 5.0],
        "false_alarm_probability": 0.01,
        "spectral_weight": 0.5,
        "p_value_decades": 3.0,
        "acf_floor": 0.2,
        "acf_good": 0.7,
        "acf_buffer_s": 20.0,
    },
    "display": {"score_history_length": 100, "score_threshold": 0.6, "full_screen": False},
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

N_PIPELINE_FRAMES = 6
"""Number of frames (after warm-up) pinned by the end-to-end case."""


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


def case_clutter() -> dict[str, np.ndarray]:
    """The four clutter-suppression modes, fed block by block."""
    x = _test_signal(6_000, 2000.0, seed=2)
    out = {}
    for mode in ("mean", "iir", "mti", "butterworth"):
        clutter_filter = ClutterFilter(
            mode=mode, f_s_hz=2000.0, alpha=0.999,
            butterworth_order=2, butterworth_cutoff_hz=0.05,
        )
        out[mode] = np.concatenate([clutter_filter(b) for b in np.array_split(x, 7)])
    return out


def case_windows() -> dict[str, np.ndarray]:
    """Every supported analysis window."""
    return {mode: get_window(mode, 64) for mode in ("hann", "hamming", "blackman", "flattop", "none")}


def case_spectral_column() -> dict[str, np.ndarray]:
    """One STFT column of a windowed segment."""
    segment = _test_signal(1024, 2000.0, seed=3)
    column = compute_single_column(segment, 2000.0, get_window("hann", 1024))
    return {"col_db": column.power_db, "f_hz": column.f_hz}


def case_detection() -> dict[str, np.ndarray]:
    """Fisher test, phase ACF, score fusion and the streaming detector."""
    n_fft, f_s = 4096, 1000.0
    f_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / f_s))
    rng = np.random.default_rng(4)
    spectrum_lin = rng.exponential(1.0, n_fft)
    spectrum_lin[np.abs(np.abs(f_hz - 244.140625) - 0.488) < 0.01] *= 50.0
    col_db = 10.0 * np.log10(spectrum_lin)

    t = np.arange(20_000) / f_s
    phi = 1.5 * np.sin(2 * np.pi * 0.3 * t) + 0.2 * rng.standard_normal(t.size)

    p_value, ratio = _fisher_p_value(spectrum_lin, f_hz, (0.1, 0.8), (2.0, 5.0), f_center_hz=244.140625)
    acf_peak, fv = _acf_peak(phi, f_s, (0.1, 0.8))
    fused = _fusion_score(p_value, acf_peak, spectral_weight=0.5, p_value_decades=3.0, acf_floor=0.2, acf_good=0.7)
    column_result = detect_presence_column(
        column_db=col_db, f_hz=f_hz, phase_rad=phi, f_s_hz=f_s,
        breathing_band_hz=(0.1, 0.8), reference_band_hz=(2.0, 5.0), f_center_hz=244.140625,
        spectral_weight=0.5, p_value_decades=3.0, acf_floor=0.2, acf_good=0.7,
    )
    return {
        "fisher": np.array([p_value, ratio]),
        "acf": np.array([acf_peak, fv]),
        "fusion": np.array([fused]),
        "column": np.array(column_result, dtype=np.float64),
    }


def case_tx_buffer() -> dict[str, np.ndarray]:
    """CW and CW-offset transmit buffers (offset = whole number of periods)."""
    return {
        "cw": cw_tx_buffer("cw", 1024, 2e6),
        "cw_offset": cw_tx_buffer("cw_offset", 16384, 2e6, 488.28125),
    }


def case_link_budget() -> dict[str, np.ndarray]:
    """Pessimistic and optimistic ranges of the radar equation."""
    return {"range_m": np.array(range_interval_m(pipeline_config()))}


def case_dashboard_context() -> dict[str, np.ndarray]:
    """Static quantities shown on the dashboard."""
    context = build_context(pipeline_config())
    return {
        "f_hz": context["f_hz"],
        "spectre_tx_db": context["tx_spectrum_db"],
        "f_hz_tx": context["tx_f_hz"],
        "scalars": np.array([
            context["f_s_dec_hz"], context["frequency_resolution_hz"],
            context["velocity_resolution_m_s"], context["range_min_m"], context["range_max_m"],
        ]),
    }


def case_pipeline() -> dict[str, np.ndarray]:
    """End-to-end streaming pipeline on the seeded simulation.

    The simulated source draws its noise from ``np.random.default_rng()``;
    the caller must make that generator deterministic (see
    ``seeded_default_rng`` in this module).
    """
    frames = streaming_frame_generator(pipeline_config(), open_source(pipeline_config(), simulation=True))
    scalars, columns = [], []
    for k, frame in zip(range(N_PIPELINE_FRAMES), frames):
        scalars.append([
            frame["frame_number"], frame["presence_score"], frame["p_value"],
            frame["acf_peak"],
            np.nan if frame["breathing_rate_hz"] is None else frame["breathing_rate_hz"],
            float(frame["alert"]),
        ])
        if k in (0, N_PIPELINE_FRAMES - 1):
            columns.append(frame["spectrum_column_db"])
    frames.close()
    return {"scalars": np.array(scalars, dtype=np.float64), "columns": np.array(columns)}


CASES: dict[str, Callable[[], dict[str, np.ndarray]]] = {
    "decimator": case_decimator,
    "clutter": case_clutter,
    "windows": case_windows,
    "spectral_column": case_spectral_column,
    "detection": case_detection,
    "tx_buffer": case_tx_buffer,
    "link_budget": case_link_budget,
    "dashboard_context": case_dashboard_context,
    "pipeline": case_pipeline,
}


class seeded_default_rng:
    """Context manager making ``np.random.default_rng()`` deterministic.

    Calls without an explicit seed get :data:`SIMULATION_SEED`; calls with a
    seed are left untouched.
    """

    def __enter__(self) -> "seeded_default_rng":
        self._original = np.random.default_rng

        def _seeded(seed=None, *args, **kwargs):
            return self._original(SIMULATION_SEED if seed is None else seed, *args, **kwargs)

        np.random.default_rng = _seeded
        return self

    def __exit__(self, *exc) -> None:
        np.random.default_rng = self._original
