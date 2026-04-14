"""Short-Time Fourier Transform (STFT) computation for micro-Doppler analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft as _scipy_stft

from MicroDopplerDetection.pipeline.windowing import get_window

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT: float = 299_792_458.0


@dataclass
class SpectrogramOutput:
    """Container for STFT results and derived physical axes.

    Attributes
    ----------
    Z : numpy.ndarray
        Complex STFT matrix, shape ``(n_freq, n_time)``, frequency axis
        centred (fftshifted).
    S_db : numpy.ndarray
        Power spectrogram in dB, ``20·log10(|Z| + eps)``, same shape.
    f_hz : numpy.ndarray
        Doppler frequency axis (Hz), centred around 0, shape ``(n_freq,)``.
    v_mps : numpy.ndarray
        Radial velocity axis (m/s), ``v = f·λ/2``, shape ``(n_freq,)``.
    t_s : numpy.ndarray
        Slow-time axis (s), shape ``(n_time,)``.
    df_hz : float
        Frequency resolution (Hz) = ``f_s / n_fft``.
    dv_mps : float
        Velocity resolution (m/s) = ``df_hz · λ / 2``.
    """

    Z: np.ndarray
    S_db: np.ndarray
    f_hz: np.ndarray
    v_mps: np.ndarray
    t_s: np.ndarray
    df_hz: float
    dv_mps: float


def compute_spectrogram(
    iq: np.ndarray,
    f_s: float,
    f_c: float,
    n_fft: int,
    overlap: float,
    window_mode: str,
) -> SpectrogramOutput:
    """Compute the STFT of a decimated, clutter-suppressed IQ signal.

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ samples (1-D) after decimation and clutter removal.
    f_s : float
        Sampling rate of *iq* (Hz), i.e. the decimated rate.
    f_c : float
        Carrier frequency (Hz), used to convert Doppler shifts to radial
        velocities via ``v = f_doppler · λ / 2``.
    n_fft : int
        FFT length (number of frequency bins), e.g. 8192.
    overlap : float
        Overlap ratio between successive STFT segments (0 to < 1),
        e.g. 0.90.
    window_mode : str
        Window name passed to :func:`~pipeline.windowing.get_window`.

    Returns
    -------
    SpectrogramOutput
        Dataclass containing the complex STFT matrix, dB spectrogram,
        frequency / velocity / time axes, and resolution figures.

    Notes
    -----
    The STFT decomposes the time-domain IQ stream into a 2-D
    time–frequency representation.  For a CW micro-Doppler radar this
    reveals **how the Doppler content evolves over slow time**:

    * Respiratory motion creates a characteristic sinusoidal trace at
      ±f_v in the spectrogram (the "micro-Doppler signature").
    * Harmonics at ±2·f_v, ±3·f_v appear due to the non-linear
      phase-modulation (Bessel expansion), though J_2, J_3 are weaker.
    * The frequency axis is converted to radial velocity using the
      Doppler relation v = f·λ/2, providing a physically meaningful
      display on the dashboard.

    The frequency axis is **centred** (via ``fftshift``) so that 0 Hz /
    0 m/s sits in the middle of the axis, with negative velocities
    (approaching target) below and positive (receding) above.
    """
    wavelength = _SPEED_OF_LIGHT / f_c
    nperseg = n_fft
    noverlap = int(n_fft * overlap)

    window = get_window(window_mode, nperseg)

    logger.info(
        "STFT — n_fft=%d, overlap=%.0f%%, fenêtre='%s', f_s=%.1f Hz",
        n_fft,
        overlap * 100,
        window_mode,
        f_s,
    )

    f_raw, t_raw, Z_raw = _scipy_stft(
        iq,
        fs=f_s,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=n_fft,
        return_onesided=False,
    )

    f_hz = np.fft.fftshift(f_raw)
    Z = np.fft.fftshift(Z_raw, axes=0)
    t_s = np.asarray(t_raw, dtype=np.float64)

    eps = 1e-12
    S_db = 20.0 * np.log10(np.abs(Z) + eps)

    v_mps = f_hz * wavelength / 2.0

    df_hz = f_s / n_fft
    dv_mps = df_hz * wavelength / 2.0

    logger.info(
        "Spectrogramme — %d bins freq × %d trames, δf=%.4f Hz, δv=%.4f m/s",
        Z.shape[0],
        Z.shape[1],
        df_hz,
        dv_mps,
    )

    return SpectrogramOutput(
        Z=Z.astype(np.complex64),
        S_db=S_db.astype(np.float64),
        f_hz=f_hz.astype(np.float64),
        v_mps=v_mps.astype(np.float64),
        t_s=t_s,
        df_hz=df_hz,
        dv_mps=dv_mps,
    )
