"""Spectral window functions for STFT analysis."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import windows as _win

logger = logging.getLogger(__name__)

_WINDOW_REGISTRY: dict[str, str] = {
    "hann": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
}


def get_window(mode: str, n: int) -> np.ndarray:
    """Return a real-valued window of length *n*.

    Parameters
    ----------
    mode : str
        Window type: ``"hann"``, ``"hamming"``, ``"blackman"``, or
        ``"none"`` (rectangular — all ones).
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
    * None (rectangular) — maximum resolution, maximum leakage; useful
      only when the signal is well-isolated in frequency.
    """
    if mode == "none":
        logger.debug("Fenêtre rectangulaire (none) — %d points", n)
        return np.ones(n, dtype=np.float64)

    if mode not in _WINDOW_REGISTRY:
        raise ValueError(
            f"Fenêtre inconnue : '{mode}'. "
            f"Utiliser 'hann', 'hamming', 'blackman' ou 'none'."
        )

    w = getattr(_win, _WINDOW_REGISTRY[mode])(n)
    logger.debug("Fenêtre '%s' — %d points", mode, n)
    return np.asarray(w, dtype=np.float64)
