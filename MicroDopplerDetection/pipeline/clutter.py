"""Static-clutter suppression filters for micro-Doppler radar."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ClutterFilter:
    """Stateful clutter-suppression filter for streaming pipelines.

    Parameters
    ----------
    mode : str
        Clutter-removal strategy: ``"mean"``, ``"iir"``, or ``"mti"``.
    alpha : float, optional
        EMA memory factor for ``"iir"`` mode.  Default is 0.99.

    Notes
    -----
    Unlike the stateless :func:`remove_clutter` function, this class
    retains its internal state (running mean for IIR, previous sample for
    MTI) across successive calls to :meth:`__call__`.  This is essential
    in a streaming pipeline where each buffer must be processed
    incrementally while maintaining filter continuity.
    """

    def __init__(self, mode: str, alpha: float = 0.99) -> None:
        if mode not in ("mean", "iir", "mti"):
            raise ValueError(
                f"Mode clutter inconnu : '{mode}'. Utiliser 'mean', 'iir' ou 'mti'."
            )
        self._mode = mode
        self._alpha = alpha
        self._mu: complex | None = None
        self._prev: complex | None = None
        logger.info(
            "ClutterFilter initialisé — mode='%s'%s",
            mode,
            f", alpha={alpha:.4f}" if mode == "iir" else "",
        )

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        """Filter one buffer of IQ samples.

        Parameters
        ----------
        iq : numpy.ndarray
            Complex IQ samples (1-D).

        Returns
        -------
        numpy.ndarray
            Clutter-suppressed IQ (same length for ``"mean"`` and
            ``"iir"``; ``len(iq)`` for ``"mti"`` — the first sample uses
            the previous buffer's last sample for continuity).
        """
        if self._mode == "mean":
            return self._apply_mean(iq)
        if self._mode == "iir":
            return self._apply_iir(iq)
        return self._apply_mti(iq)

    def _apply_mean(self, iq: np.ndarray) -> np.ndarray:
        """Subtract the block mean."""
        return (iq - np.mean(iq)).astype(iq.dtype)

    def _apply_iir(self, iq: np.ndarray) -> np.ndarray:
        """Recursive EMA high-pass with state carry-over."""
        out = np.empty_like(iq)
        mu = self._mu if self._mu is not None else complex(iq[0])
        alpha = self._alpha
        for n in range(len(iq)):
            mu = alpha * mu + (1.0 - alpha) * iq[n]
            out[n] = iq[n] - mu
        self._mu = mu
        return out

    def _apply_mti(self, iq: np.ndarray) -> np.ndarray:
        """Single-delay MTI with state carry-over."""
        prev = self._prev if self._prev is not None else complex(iq[0])
        extended = np.concatenate(([prev], iq))
        self._prev = complex(iq[-1])
        return np.diff(extended)


# ======================================================================
# Stateless convenience function (backward compatible)
# ======================================================================

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
