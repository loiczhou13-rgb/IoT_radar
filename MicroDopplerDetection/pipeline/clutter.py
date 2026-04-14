"""Static-clutter suppression filters for micro-Doppler radar."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def remove_clutter(
    iq: np.ndarray,
    mode: str,
    alpha: float = 0.99,
) -> np.ndarray:
    """Remove or attenuate the static clutter (DC component) from *iq*.

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ samples (1-D), typically after decimation.
    mode : str
        Clutter-removal strategy:

        * ``"mean"`` — subtract the global mean of *iq*.
        * ``"iir"``  — recursive EMA high-pass filter with memory factor
          *alpha*.
        * ``"mti"``  — single-delay Moving Target Indicator (first-order
          difference).
    alpha : float, optional
        EMA memory factor for ``"iir"`` mode (0 < alpha < 1).
        Closer to 1 → longer memory → lower cut-off frequency.
        Default is 0.99.

    Returns
    -------
    numpy.ndarray
        Clutter-suppressed IQ of same shape as *iq* (or ``len(iq)-1`` for
        ``"mti"``).

    Raises
    ------
    ValueError
        If *mode* is not one of ``{"mean", "iir", "mti"}``.

    Notes
    -----
    In a CW micro-Doppler radar the dominant return is **static clutter**:
    direct TX→RX leakage, reflections from walls, rubble, and ground.
    All of these appear at (or very near) 0 Hz in baseband.  The
    respiratory signal at ±f_v is 40–60 dB weaker, so effective clutter
    suppression is essential before spectral analysis.

    * **Mean subtraction** is a batch operation — simple but assumes the
      clutter is time-invariant.
    * **IIR / EMA** tracks slow clutter drift adaptively:
      μ[n] = α·μ[n−1] + (1−α)·x[n],  y[n] = x[n] − μ[n].
      Approximate high-pass cut-off: f_hp ≈ (1−α)·f_s / (2π).
    * **MTI** (y[n] = x[n] − x[n−1]) provides first-order cancellation
      of any constant component, acting as a comb-null at 0 Hz.
    """
    if mode not in ("mean", "iir", "mti"):
        raise ValueError(
            f"Mode clutter inconnu : '{mode}'. Utiliser 'mean', 'iir' ou 'mti'."
        )

    if mode == "mean":
        logger.info("Suppression du clutter — soustraction de la moyenne globale")
        return (iq - np.mean(iq)).astype(iq.dtype)

    if mode == "iir":
        logger.info(
            "Suppression du clutter — filtre IIR/EMA (alpha = %.4f)", alpha
        )
        return _ema_highpass(iq, alpha)

    logger.info("Suppression du clutter — annulateur MTI (np.diff)")
    return np.diff(iq)


def _ema_highpass(iq: np.ndarray, alpha: float) -> np.ndarray:
    """Apply a recursive EMA high-pass filter to *iq*."""
    out = np.empty_like(iq)
    mu = iq[0].copy()
    for n in range(len(iq)):
        mu = alpha * mu + (1.0 - alpha) * iq[n]
        out[n] = iq[n] - mu
    return out
