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

from MicroDopplerDetection.main import (
    _build_context,
    _compute_range,
    _streaming_frame_generator,
)
from MicroDopplerDetection.pipeline.clutter import ClutterFilter
from MicroDopplerDetection.pipeline.decimation import Decimator
from MicroDopplerDetection.pipeline.detection import (
    _acf_peak,
    _fisher_pvalue,
    _fusion_score,
    detect_presence_column,
)
from MicroDopplerDetection.pipeline.emission import generate_tx_buffer
from MicroDopplerDetection.pipeline.spectrogramme import compute_single_column
from MicroDopplerDetection.pipeline.windowing import get_window

SIMULATION_SEED = 1234
"""Seed of the random generator used by the simulated IQ source."""

# Reduced configuration (fast) of the streaming pipeline.  The TX offset
# 244.140625 Hz is exactly 10 periods of a 4096-sample buffer at 100 kHz, so
# the cyclic TX buffer is continuous (see bug B1 in REFACTOR_PLAN.md).
PIPELINE_CONFIG: dict[str, Any] = {
    "logging": {"level": "WARNING", "to_file": False},
    "sdr": {
        "uri": "ip:192.168.2.1",
        "f_c": 3.5e9,
        "f_s": 100e3,
        "rx_gain": 45,
        "tx_gain": -20,
        "buffer_size": 4096,
    },
    "emission": {"mode": "cw_offset", "f_offset": 244.140625},
    "decimation": {"enable": True, "D": 100, "f_max_utile": 10},
    "clutter": {
        "mode": "butterworth",
        "alpha": 0.9999,
        "butterworth_order": 2,
        "butterworth_cutoff": 0.05,
    },
    "windowing": {"mode": "hann"},
    "spectrogramme": {"n_fft": 4096, "overlap": 0.90, "skip_warmup": 2},
    "detection": {
        "bande_respiration": [0.1, 0.8],
        "bande_reference": [2.0, 5.0],
        "alpha": 0.01,
        "w": 0.5,
        "p_value_decades": 3.0,
        "acf_floor": 0.2,
        "acf_good": 0.7,
        "acf_buffer_seconds": 20.0,
    },
    "affichage": {"N_historique": 100, "seuil_proba": 0.6, "plein_ecran": False},
    "bilan_liaison": {
        "optimiste": {
            "P_tx_dBm": -13, "G_tx_dBi": 2, "G_rx_dBi": 2, "sigma_m2": 0.5,
            "NF_dB": 4, "L_sys_dB": 3, "SNR_min_dB": 3,
        },
        "pessimiste": {
            "P_tx_dBm": -13, "G_tx_dBi": 0, "G_rx_dBi": 0, "sigma_m2": 0.05,
            "NF_dB": 6, "L_sys_dB": 25, "SNR_min_dB": 3,
        },
        "B_eff_hz": 50,
    },
    "simulation": {
        "enable": True,
        "fv": 0.3,
        "D_mm": 10,
        "snr_dB": 20,
        "clutter_amplitude": 100.0,
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
        decimator = Decimator(f_s=2e6, D=factor, f_max_utile=10.0)
        blocks = np.array_split(x, 13)
        out[f"d{factor}"] = np.concatenate([decimator(b) for b in blocks])
    return out


def case_clutter() -> dict[str, np.ndarray]:
    """The four clutter-suppression modes, fed block by block."""
    x = _test_signal(6_000, 2000.0, seed=2)
    out = {}
    for mode in ("mean", "iir", "mti", "butterworth"):
        clutter_filter = ClutterFilter(
            mode=mode, fs=2000.0, alpha=0.999,
            butterworth_order=2, butterworth_cutoff=0.05,
        )
        out[mode] = np.concatenate([clutter_filter(b) for b in np.array_split(x, 7)])
    return out


def case_windows() -> dict[str, np.ndarray]:
    """Every supported analysis window."""
    return {mode: get_window(mode, 64) for mode in ("hann", "hamming", "blackman", "flattop", "none")}


def case_spectral_column() -> dict[str, np.ndarray]:
    """One STFT column of a windowed segment."""
    segment = _test_signal(1024, 2000.0, seed=3)
    column = compute_single_column(segment, 2000.0, 3.5e9, get_window("hann", 1024))
    return {"col_db": column.col_db, "f_hz": column.f_hz}


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

    p_value, ratio = _fisher_pvalue(spectrum_lin, f_hz, (0.1, 0.8), (2.0, 5.0), f_center=244.140625)
    acf_peak, fv = _acf_peak(phi, f_s, (0.1, 0.8))
    fused = _fusion_score(p_value, acf_peak, w=0.5, p_value_decades=3.0, acf_floor=0.2, acf_good=0.7)
    column_result = detect_presence_column(
        col_db=col_db, f_hz=f_hz, phi_buffer=phi, f_s=f_s,
        bande_respiration=(0.1, 0.8), bande_reference=(2.0, 5.0), f_center=244.140625,
        w=0.5, p_value_decades=3.0, acf_floor=0.2, acf_good=0.7,
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
        "cw": generate_tx_buffer("cw", 1024, 2e6),
        "cw_offset": generate_tx_buffer("cw_offset", 16384, 2e6, 488.28125),
    }


def case_link_budget() -> dict[str, np.ndarray]:
    """Pessimistic and optimistic ranges of the radar equation."""
    return {"range_m": np.array(_compute_range(pipeline_config()))}


def case_dashboard_context() -> dict[str, np.ndarray]:
    """Static quantities shown on the dashboard."""
    context = _build_context(pipeline_config())
    return {
        "f_hz": context["f_hz"],
        "spectre_tx_db": context["spectre_tx_db"],
        "f_hz_tx": context["f_hz_tx"],
        "scalars": np.array([
            context["f_s_dec"], context["df_hz"], context["dv_mps"],
            context["R_min_m"], context["R_max_m"],
        ]),
    }


def case_pipeline() -> dict[str, np.ndarray]:
    """End-to-end streaming pipeline on the seeded simulation.

    The simulated source draws its noise from ``np.random.default_rng()``;
    the caller must make that generator deterministic (see
    ``seeded_default_rng`` in this module).
    """
    frames = _streaming_frame_generator(pipeline_config(), simulation=True)
    scalars, columns = [], []
    for k, frame in zip(range(N_PIPELINE_FRAMES), frames):
        scalars.append([
            frame["n_trame"], frame["score_presence"], frame["p_value_f"],
            frame["acf_peak"],
            np.nan if frame["fv_estimated"] is None else frame["fv_estimated"],
            float(frame["detection"]),
        ])
        if k in (0, N_PIPELINE_FRAMES - 1):
            columns.append(frame["spectre_colonne"])
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
