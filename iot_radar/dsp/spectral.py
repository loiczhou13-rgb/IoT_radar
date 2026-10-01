"""Spectral analysis helpers: analysis windows and single STFT columns.

The streaming pipeline computes one STFT column at a time with
:func:`compute_single_column`; the window applied to each segment comes from
:func:`get_window`.  (Merged from the former ``pipeline/windowing.py`` and
``pipeline/spectrogramme.py``.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import windows

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT: float = 299_792_458.0

_VALID_WINDOWS = ("hann", "hamming", "blackman", "flattop")


# ---------------------------------------------------------------------------
# Analysis windows
# ---------------------------------------------------------------------------

def get_window(mode: str, n: int) -> np.ndarray:
    """Return a real-valued window of length *n*.

    Parameters
    ----------
    mode : str
        Window type: ``"hann"``, ``"hamming"``, ``"blackman"``,
        ``"flattop"``, or ``"none"`` (rectangular — all ones).
    n : int
        Number of samples in the window.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(n,)`` with values in [0, 1].

    Raises
    ------
    ValueError
        If *mode* is not a recognised window name.

    Notes
    -----
    In STFT-based micro-Doppler analysis each time segment is multiplied
    by a tapering window before the FFT.  This reduces **spectral leakage**
    — energy from strong clutter residuals or harmonics spilling into the
    weak respiratory bins.  The trade-off is a wider main lobe (lower
    frequency resolution).

    * Hann — good general-purpose choice; −31 dB first sidelobe.
    * Hamming — slightly lower sidelobes (−43 dB) at the cost of a
      discontinuity at the edges.
    * Blackman — very low sidelobes (−58 dB), ~50 % wider main lobe.
    * Flat-top — best amplitude accuracy (< 0.01 dB error), useful for
      measuring Bessel-series harmonic magnitudes, but widest main lobe
      (~3.8× Hann) — not recommended for detection where frequency
      resolution is critical.
    * None (rectangular) — maximum resolution, maximum leakage; useful
      only when the signal is well-isolated in frequency.
    """
    if mode == "none":
        logger.debug("Fenêtre rectangulaire (none) — %d points", n)
        return np.ones(n, dtype=np.float64)

    if mode not in _VALID_WINDOWS:
        raise ValueError(
            f"Fenêtre inconnue : '{mode}'. "
            f"Utiliser {', '.join(repr(m) for m in _VALID_WINDOWS)} ou 'none'."
        )

    w = getattr(windows, mode)(n)
    logger.debug("Fenêtre '%s' — %d points", mode, n)
    return np.asarray(w, dtype=np.float64)


# ---------------------------------------------------------------------------
# Single STFT column
# ---------------------------------------------------------------------------

@dataclass
class ColumnOutput:
    """Container for a single STFT column (one spectral snapshot).

    Attributes
    ----------
    col_complex : numpy.ndarray
        Complex spectrum (complex128), shape ``(n_fft,)``, fftshifted.
    col_db : numpy.ndarray
        Power in dB, shape ``(n_fft,)``.
    f_hz : numpy.ndarray
        Doppler frequency axis (Hz), centred.
    v_mps : numpy.ndarray
        Radial velocity axis (m/s).
    df_hz : float
        Frequency resolution (Hz).
    """

    col_complex: np.ndarray
    col_db: np.ndarray
    f_hz: np.ndarray
    v_mps: np.ndarray
    df_hz: float


def compute_single_column(
    segment: np.ndarray,
    f_s: float,
    f_c: float,
    window: np.ndarray,
) -> ColumnOutput:
    """Compute one STFT column from a windowed time-domain segment.

    Parameters
    ----------
    segment : numpy.ndarray
        Complex IQ segment, shape ``(n_fft,)``.
    f_s : float
        Sampling rate (Hz) of *segment* (decimated rate).
    f_c : float
        Carrier frequency (Hz), for Doppler-to-velocity conversion.
    window : numpy.ndarray
        Pre-computed window, same length as *segment*.

    Returns
    -------
    ColumnOutput
        Single spectrum column with frequency / velocity axes.
    """
    n_fft = len(segment)
    wavelength = _SPEED_OF_LIGHT / f_c

    windowed = segment * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))

    eps = 1e-12
    col_db = 10.0 * np.log10(np.abs(spectrum) ** 2 + eps)

    f_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / f_s))
    v_mps = f_hz * wavelength / 2.0
    df_hz = f_s / n_fft

    return ColumnOutput(
        col_complex=spectrum,
        col_db=col_db.astype(np.float64),
        f_hz=f_hz.astype(np.float64),
        v_mps=v_mps.astype(np.float64),
        df_hz=df_hz,
    )
