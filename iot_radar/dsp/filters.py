"""Window-based filters of the displacement signal.

Place in the chain: after the phase demodulation.  The analysis works on a
sliding window (20 s by default), so **zero-phase** filtering
(``sosfiltfilt``) is legitimate here and does not distort the breathing
waveform.  (The causal, stateful filtering of the raw stream is done by
:class:`iot_radar.dsp.decimation.Decimator`.)
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
    """Zero-phase Butterworth band-pass filter of a real window.

    Parameters
    ----------
    x : numpy.ndarray
        Real 1-D signal (e.g. chest displacement in mm).
    f_s_hz : float
        Sampling rate of *x* (Hz).
    band_hz : tuple[float, float]
        Pass band ``(low, high)`` in Hz, with ``0 < low < high < f_s / 2``.
    order : int, optional
        Butterworth order of each edge (the band-pass has order ``2·order``,
        doubled again by the forward-backward filtering).

    Returns
    -------
    numpy.ndarray
        Filtered signal, same length and unit as *x*.
    """
    low_hz, high_hz = float(band_hz[0]), float(band_hz[1])
    if not 0.0 < low_hz < high_hz < f_s_hz / 2.0:
        raise ValueError(f"Invalid band {band_hz} for f_s={f_s_hz} Hz.")
    sos = butter(order, (low_hz, high_hz), btype="bandpass", fs=f_s_hz, output="sos")
    x = np.asarray(x, dtype=np.float64)
    # A long odd-symmetric extension (the whole window) lets the slow
    # high-pass edge settle; the scipy default (~27 samples) leaves strong
    # edge transients on sub-Hz bands.
    return sosfiltfilt(sos, x, padtype="odd", padlen=max(x.size - 1, 0))


def widened_band(
    band_hz: tuple[float, float],
    f_s_hz: float,
    factor: float = 1.5,
) -> tuple[float, float]:
    """Filter band slightly wider than a search band.

    ``sosfiltfilt`` squares the magnitude response, so a Butterworth band-pass
    designed exactly on the search band would attenuate its edges by 6 dB.
    The filter is designed on ``(low / factor, high · factor)``, the upper
    edge being kept below 0.45 × the sampling rate.
    """
    return (band_hz[0] / factor, min(band_hz[1] * factor, 0.45 * f_s_hz))


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove the least-squares straight line of a real window.

    Removes the slow phase drift (residual LO offset, thermal drift) before a
    spectral analysis.
    """
    return _scipy_detrend(np.asarray(x, dtype=np.float64), type="linear")
