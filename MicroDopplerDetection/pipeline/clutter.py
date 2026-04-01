"""
Clutter suppression for zero-Doppler and slowly varying static returns.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ClutterMode = Literal["mean", "iir", "mti"]


def suppress_clutter(x: np.ndarray, mode: ClutterMode, alpha: float = 0.99) -> np.ndarray:
    """
    Remove strong static clutter using mean removal, EMA tracking, or MTI delay.

    Parameters
    ----------
    x
        Complex IQ after decimation, shape ``(N,)``.
    mode
        ``"mean"``: subtract global mean (DC and static average phasor).
        ``"iir"``: exponential moving average clutter tracker per sample.
        ``"mti"``: first-order difference along time (two-pulse canceler).
    alpha
        EMA pole for ``"iir"`` mode, in ``[0.9, 0.999]`` (typical ``0.99``).

    Returns
    -------
    numpy.ndarray
        Filtered IQ, same shape as ``x`` (MTI shortens by one sample — we pad
        the first sample with ``0`` to preserve length).

    Examples
    --------
    >>> x = np.ones(100, dtype=np.complex64)
    >>> suppress_clutter(x, "mean")[0]
    (0+0j)

    Notes
    -----
    Physical note: walls, floor, and stationary debris dominate at 0 Hz;
    subtracting a slow clutter estimate exposes the weak breathing-induced
    sidebands that sit only a fraction of a hertz away from DC after decimation.
    """
    x = np.asarray(x, dtype=np.complex64)
    if x.ndim != 1:
        raise ValueError("suppress_clutter expects a 1-D array")

    if mode == "mean":
        mu = np.mean(x)
        y = x - mu
        logger.debug("Clutter mean removal: |mean|=%.4g", float(np.abs(mu)))

    elif mode == "iir":
        if not (0.9 <= alpha <= 0.999):
            logger.warning("alpha=%g is outside [0.9, 0.999]; proceeding anyway", alpha)
        c_hat = np.zeros_like(x, dtype=np.complex128)
        y = np.zeros_like(x, dtype=np.complex128)
        a = float(alpha)
        for n in range(x.shape[0]):
            if n == 0:
                c_hat[n] = (1.0 - a) * x[n]
            else:
                c_hat[n] = a * c_hat[n - 1] + (1.0 - a) * x[n]
            y[n] = x[n] - c_hat[n]
        y = y.astype(np.complex64)
        logger.debug("Clutter IIR: alpha=%g", a)

    elif mode == "mti":
        # x_f[n] = x[n] - x[n-1]; x_f[0] = 0 to keep length for STFT alignment
        y = np.zeros_like(x, dtype=np.complex64)
        y[1:] = x[1:] - x[:-1]
        logger.debug("Clutter MTI (first difference) applied")

    else:
        raise ValueError(f"Unknown clutter mode: {mode!r}")

    return y
