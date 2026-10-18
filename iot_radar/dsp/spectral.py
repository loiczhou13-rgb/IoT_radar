"""Analysis windows and STFT columns of the micro-Doppler waterfall (display only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import windows

logger = logging.getLogger(__name__)

_VALID_WINDOWS = ("hann", "hamming", "blackman", "flattop")


# ---------------------------------------------------------------------------
# Analysis windows
# ---------------------------------------------------------------------------

def get_window(window_name: str, n_samples: int) -> np.ndarray:
    """Analysis window of *n_samples* points: ``"hann"``, ``"hamming"``,
    ``"blackman"``, ``"flattop"`` or ``"none"`` (rectangular)."""
    if window_name == "none":
        logger.debug("Rectangular window (none) — %d points", n_samples)
        return np.ones(n_samples, dtype=np.float64)

    if window_name not in _VALID_WINDOWS:
        raise ValueError(
            f"Unknown window '{window_name}'. "
            f"Use {', '.join(repr(m) for m in _VALID_WINDOWS)} or 'none'."
        )

    w = getattr(windows, window_name)(n_samples)
    logger.debug("Window '%s' — %d points", window_name, n_samples)
    return np.asarray(w, dtype=np.float64)


# ---------------------------------------------------------------------------
# Single STFT column
# ---------------------------------------------------------------------------

def frequency_axis(n_fft: int, f_s_hz: float) -> np.ndarray:
    """Frequency axis (Hz) of an fftshifted *n_fft*-point spectrum, from ``−f_s/2``."""
    frequencies = np.fft.fftfreq(n_fft, d=1.0 / f_s_hz)
    return np.fft.fftshift(frequencies).astype(np.float64)


@dataclass
class ColumnOutput:
    """One STFT column: power (dB, fftshifted) and its frequency axis (Hz)."""

    power_db: np.ndarray
    f_hz: np.ndarray


def compute_single_column(
    segment: np.ndarray,
    f_s_hz: float,
    window: np.ndarray,
    n_fft: int | None = None,
) -> ColumnOutput:
    """Power spectrum (dB) of one windowed complex segment, zero-padded to *n_fft*."""
    n_fft = len(segment) if n_fft is None else int(n_fft)

    windowed = segment * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))

    eps = 1e-12
    power_db = 10.0 * np.log10(np.abs(spectrum) ** 2 + eps)

    return ColumnOutput(
        power_db=power_db.astype(np.float64),
        f_hz=frequency_axis(n_fft, f_s_hz),
    )
