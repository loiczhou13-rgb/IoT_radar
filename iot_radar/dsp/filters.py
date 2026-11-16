"""Filters of the displacement, on a whole analysis window (after the demodulation).

The window is complete, so the filters can be zero-phase (``sosfiltfilt``).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend as _scipy_detrend, sosfiltfilt


def bandpass(
    x: np.ndarray,
    f_s_hz: float,
    band_hz: tuple[float, float],
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass of a real window (same length and unit)."""
    low_hz, high_hz = float(band_hz[0]), float(band_hz[1])
    if not 0.0 < low_hz < high_hz < f_s_hz / 2.0:
        raise ValueError(f"Invalid band {band_hz} for f_s={f_s_hz} Hz.")
    sos = butter(order, (low_hz, high_hz), btype="bandpass", fs=f_s_hz, output="sos")
    x = np.asarray(x, dtype=np.float64)
    # Padding over the whole window: the scipy default (~27 samples) leaves
    # edge transients on sub-Hz bands.
    return sosfiltfilt(sos, x, padtype="odd", padlen=max(x.size - 1, 0))


def widened_band(
    band_hz: tuple[float, float],
    f_s_hz: float,
    factor: float = 1.5,
) -> tuple[float, float]:
    """``(low / factor, high · factor)``, top below 0.45·f_s.

    ``sosfiltfilt`` squares the response: a filter designed on the search
    band itself would cut its edges by 6 dB.
    """
    return (band_hz[0] / factor, min(band_hz[1] * factor, 0.45 * f_s_hz))


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove the least-squares straight line (slow drift) of a real window."""
    return _scipy_detrend(np.asarray(x, dtype=np.float64), type="linear")
