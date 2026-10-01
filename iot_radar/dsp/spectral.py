"""Spectral analysis: analysis windows, STFT columns and spectrograms.

Place in the chain: visualisation only — nothing here feeds the breathing
decision.  The pipeline computes one STFT column of the slow-time IQ at each
update (:func:`compute_single_column`) for the micro-Doppler waterfall of the
dashboard; a complete signal (e.g. a replayed session) can be turned offline
into a spectrogram framed the same way (:func:`compute_spectrogram`).
"""

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
    """Return a real-valued analysis window.

    Parameters
    ----------
    window_name : str
        ``"hann"``, ``"hamming"``, ``"blackman"``, ``"flattop"``, or
        ``"none"`` (rectangular — all ones).
    n_samples : int
        Length of the window.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(n_samples,)`` with values in [0, 1]
        (symmetric windows of ``scipy.signal.windows``).

    Raises
    ------
    ValueError
        If *window_name* is not a recognised window.

    Notes
    -----
    Each STFT segment is multiplied by a tapering window before the FFT.
    This reduces **spectral leakage** — energy of strong clutter residuals
    spilling into the weak breathing bins — at the cost of a wider main
    lobe (lower frequency resolution).

    * Hann — good general-purpose choice; −31 dB first sidelobe.
    * Hamming — lower sidelobes (−43 dB), discontinuous at the edges.
    * Blackman — very low sidelobes (−58 dB), ~50 % wider main lobe.
    * Flat-top — best amplitude accuracy (< 0.01 dB error) but the widest
      main lobe (~3.8× Hann): not recommended for detection.
    * None (rectangular) — best resolution, most leakage; only for
      well-isolated signals.
    """
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
    """Centred frequency axis (Hz) of an fftshifted ``n_fft``-point spectrum.

    Parameters
    ----------
    n_fft : int
        Number of FFT points.
    f_s_hz : float
        Sampling rate (Hz) of the transformed signal.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(n_fft,)`` from ``-f_s/2`` up to
        ``f_s/2 - f_s/n_fft``, in steps of ``f_s / n_fft``.
    """
    frequencies = np.fft.fftfreq(n_fft, d=1.0 / f_s_hz)
    return np.fft.fftshift(frequencies).astype(np.float64)


@dataclass
class ColumnOutput:
    """One STFT column (spectral snapshot of one segment).

    Attributes
    ----------
    power_db : numpy.ndarray
        Power ``10·log10(|X|² + 1e-12)`` (dB), shape ``(n_fft,)``, fftshifted.
    f_hz : numpy.ndarray
        Matching frequency axis (Hz), centred (see :func:`frequency_axis`).
    """

    power_db: np.ndarray
    f_hz: np.ndarray


def compute_single_column(
    segment: np.ndarray,
    f_s_hz: float,
    window: np.ndarray,
    n_fft: int | None = None,
) -> ColumnOutput:
    """Power spectrum of one windowed time segment (one STFT column).

    Parameters
    ----------
    segment : numpy.ndarray
        Complex IQ segment, 1-D.
    f_s_hz : float
        Sampling rate (Hz) of *segment* (decimated rate).
    window : numpy.ndarray
        Pre-computed window, same length as *segment*.
    n_fft : int or None, optional
        FFT size; ``None`` (default) = ``len(segment)``.  A larger size
        zero-pads the segment (interpolated spectrum, same resolution).

    Returns
    -------
    ColumnOutput
        Power in dB and its frequency axis, ``n_fft`` points.  No
        normalisation by the window sum is applied.
    """
    n_fft = len(segment) if n_fft is None else int(n_fft)

    windowed = segment * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))

    eps = 1e-12
    power_db = 10.0 * np.log10(np.abs(spectrum) ** 2 + eps)

    return ColumnOutput(
        power_db=power_db.astype(np.float64),
        f_hz=frequency_axis(n_fft, f_s_hz),
    )


# ---------------------------------------------------------------------------
# Whole spectrogram (offline)
# ---------------------------------------------------------------------------

def compute_spectrogram(
    iq: np.ndarray,
    f_s_hz: float,
    window: np.ndarray,
    hop: int,
    skip_frames: int = 0,
) -> np.ndarray:
    """Spectrogram of a complete signal, framed exactly like the stream.

    A stream framed with a buffer that, whenever it holds ``n_fft`` samples,
    turns the first ``n_fft`` of them into one column and drops the first
    ``hop`` has segments starting at sample ``k * hop`` (k = 0, 1, 2, ...).
    This function cuts the whole array the same way, so its columns are
    identical to the streamed ones.

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ samples (1-D) at rate *f_s_hz* — e.g. the slow-time
        signal.
    f_s_hz : float
        Sampling rate of *iq* (Hz).
    window : numpy.ndarray
        Analysis window; its length is the FFT size ``n_fft``.
    hop : int
        Step between the starts of two successive segments (samples).
    skip_frames : int, optional
        Number of initial columns to drop (e.g. during the warm-up of the
        decimation filters).

    Returns
    -------
    numpy.ndarray
        Power in dB, shape ``(n_frames, n_fft)``; row ``k`` is the column of
        the ``(k + skip_frames)``-th segment.  Each column is fftshifted (see
        :func:`frequency_axis` for the frequency axis).
    """
    n_fft = len(window)
    columns: list[np.ndarray] = []
    frame_number = 0
    start = 0
    while start + n_fft <= len(iq):
        frame_number += 1
        if frame_number > skip_frames:
            segment = iq[start:start + n_fft]
            columns.append(compute_single_column(segment, f_s_hz, window).power_db)
        start += hop
    return np.array(columns, dtype=np.float64).reshape(len(columns), n_fft)
