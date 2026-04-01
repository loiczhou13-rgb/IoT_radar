"""
FFT window utilities to control spectral leakage on finite segments.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

WindowMode = Literal["hann", "hamming", "blackman", "none"]


def get_window(mode: WindowMode, n_fft: int) -> np.ndarray:
    """
    Return a window vector of length ``n_fft`` for STFT/FFT framing.

    Parameters
    ----------
    mode
        ``"hann"``, ``"hamming"``, ``"blackman"``, or ``"none"`` (rectangular).
    n_fft
        Window length in samples (typical ``1024``).

    Returns
    -------
    numpy.ndarray
        Real float64 window of shape ``(n_fft,)``. For ``"none"``, all ones.

    Examples
    --------
    >>> w = get_window("hann", 16)
    >>> w.shape
    (16,)

    Notes
    -----
    Physical / DSP note: truncating slow-time IQ to a finite block is equivalent
    to multiplying by a rectangle, which convolves the true spectrum with a
    sinc-like kernel and leaks energy across Doppler bins. A tapered window
    widens the main lobe slightly but strongly suppresses sidelobes from strong
    static clutter remnants, improving detectability of weak breathing lines.
    """
    n_fft = int(n_fft)
    if n_fft < 1:
        raise ValueError("n_fft must be >= 1")

    if mode == "none":
        w = np.ones(n_fft, dtype=np.float64)
    elif mode == "hann":
        w = np.hanning(n_fft).astype(np.float64)
    elif mode == "hamming":
        w = np.hamming(n_fft).astype(np.float64)
    elif mode == "blackman":
        w = np.blackman(n_fft).astype(np.float64)
    else:
        raise ValueError(f"Unknown window mode: {mode!r}")

    logger.debug("Built %s window, n_fft=%d", mode, n_fft)
    return w
