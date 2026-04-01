"""
Short-time Fourier transform and log-magnitude spectrogram for micro-Doppler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .windowing import WindowMode, get_window

logger = logging.getLogger(__name__)


@dataclass
class SpectrogramOutput:
    """STFT products and derived axes."""

    Z: np.ndarray
    """Complex STFT, shape ``(n_fft, n_frames_stft)``."""

    S_db: np.ndarray
    """Log magnitude ``20*log10(|Z|+eps)``, same shape as ``Z``."""

    f_hz: np.ndarray
    """Doppler frequency axis (Hz), shape ``(n_fft,)``, fft-shifted."""

    t_s: np.ndarray
    """Time axis of STFT columns (s), shape ``(n_frames_stft,)``."""

    v_mps: np.ndarray
    """Velocity axis ``f * lambda / 2`` (m/s), same as ``f_hz`` mapping."""

    df_hz: float
    """Approximate frequency bin spacing (Hz)."""

    dv_mps: float
    """Approximate velocity bin spacing (m/s)."""


def compute_spectrogram(
    x: np.ndarray,
    f_s: float,
    f_c: float,
    n_fft: int,
    overlap_ratio: float,
    window_mode: WindowMode,
) -> SpectrogramOutput:
    """
    Compute the complex STFT and a decibel spectrogram for IQ slow-time.

    Parameters
    ----------
    x
        Complex IQ after clutter filtering, shape ``(N,)``.
    f_s
        Sample rate after decimation (unit: Hz).
    f_c
        Carrier frequency to compute wavelength (unit: Hz).
    n_fft
        FFT length / STFT frame length (samples, typical ``1024``).
    overlap_ratio
        Fraction of ``n_fft`` overlapped between adjacent windows (typical
        ``0.90`` → ``noverlap = int(0.9 * n_fft)``).
    window_mode
        Taper applied per segment before FFT (see :mod:`windowing`).

    Returns
    -------
    SpectrogramOutput
        ``Z``, ``S_db``, shifted frequency/velocity axes, and resolutions.

    Examples
    --------
    >>> x = np.exp(1j * 2 * np.pi * 0.3 * np.arange(8000) / 200).astype(np.complex64)
    >>> out = compute_spectrogram(x, 200.0, 2.4e9, 256, 0.5, "hann")
    >>> out.Z.ndim
    2

    Notes
    -----
    Physical note: the STFT separates slow-time variations of the echo phase
    into Doppler bins; breathing sidebands appear as narrow ridges that persist
    across time columns, whereas wideband noise fills the panel uniformly.
    """
    x = np.asarray(x, dtype=np.complex64)
    n_fft = int(n_fft)
    if n_fft < 2:
        raise ValueError("n_fft must be >= 2")
    n_in = int(x.shape[0])
    if n_in < n_fft:
        logger.warning(
            "Clamping n_fft from %d to signal length %d (capture more frames or reduce decimation D).",
            n_fft,
            n_in,
        )
        n_fft = max(2, n_in)
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")

    noverlap = int(round(overlap_ratio * n_fft))
    noverlap = min(max(noverlap, 0), n_fft - 1)

    win = get_window(window_mode, n_fft)
    nfft = n_fft

    f_raw, t_raw, Zxx = signal.stft(
        x,
        fs=f_s,
        window=win,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary="zeros",
        padded=True,
        axis=-1,
    )

    # scipy returns Zxx shape (n_freq, n_segments); for complex input n_freq == n_fft
    Z = np.asarray(Zxx, dtype=np.complex64)
    f_stft = np.asarray(f_raw, dtype=np.float64)
    t_stft = np.asarray(t_raw, dtype=np.float64)

    f_hz = np.fft.fftshift(f_stft)
    Z = np.fft.fftshift(Z, axes=0)

    c_light = 299792458.0
    wavelength = c_light / float(f_c)
    v_mps = f_hz * wavelength / 2.0

    eps = 1e-12
    S_db = (20.0 * np.log10(np.abs(Z) + eps)).astype(np.float64)

    df_hz = float(f_s) / float(n_fft)
    dv_mps = df_hz * wavelength / 2.0

    logger.info(
        "STFT: n_fft=%d overlap=%.3f -> df≈%.4g Hz, dv≈%.4g m/s, %d time frames",
        n_fft,
        overlap_ratio,
        df_hz,
        dv_mps,
        Z.shape[1],
    )

    return SpectrogramOutput(
        Z=Z,
        S_db=S_db,
        f_hz=f_hz,
        t_s=t_stft,
        v_mps=v_mps,
        df_hz=df_hz,
        dv_mps=dv_mps,
    )
