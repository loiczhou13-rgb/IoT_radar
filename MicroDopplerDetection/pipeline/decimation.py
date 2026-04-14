"""Sample-rate reduction through low-pass filtering and down-sampling."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import decimate as _scipy_decimate

logger = logging.getLogger(__name__)


def decimate_iq(
    iq: np.ndarray,
    f_s: float,
    D: int,
    f_max_utile: float,
) -> tuple[np.ndarray, float]:
    """Decimate the IQ vector by factor *D* after anti-alias filtering.

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ samples at rate *f_s*.
    f_s : float
        Current sampling rate (Hz), e.g. 2.0e6.
    D : int
        Decimation factor.  The output rate is ``f_s / D``.
    f_max_utile : float
        Maximum frequency of interest (Hz), e.g. 10.  Used to verify
        the Nyquist criterion after decimation.

    Returns
    -------
    iq_decimated : numpy.ndarray
        Decimated complex IQ vector of shape ``(len(iq) // D,)``.
    f_s_new : float
        New sampling rate (Hz) = ``f_s / D``.

    Raises
    ------
    ValueError
        If the Shannon–Nyquist criterion is violated after decimation,
        i.e. ``f_s / D <= 2 * f_max_utile``.

    Notes
    -----
    The PlutoSDR delivers IQ data at rates ≥ 521 kHz, whereas the
    respiratory micro-Doppler signal lives below a few Hz.  Decimation
    reduces the data rate by a factor *D* (typically 1000), making
    spectral analysis tractable.

    ``scipy.signal.decimate`` applies a Chebyshev type-I anti-aliasing
    low-pass filter before down-sampling, preventing high-frequency noise
    from folding into the band of interest.

    I and Q channels are decimated independently to preserve the analytic
    (complex) nature of the signal.
    """
    f_s_new = f_s / D

    if f_s_new <= 2.0 * f_max_utile:
        raise ValueError(
            f"Critère de Shannon violé après décimation : "
            f"f_s_new = {f_s_new:.1f} Hz ≤ 2 × f_max_utile = {2.0 * f_max_utile:.1f} Hz. "
            f"Réduire D (actuellement {D}) ou augmenter f_s."
        )

    logger.debug(
        "Décimation ×%d — f_s : %.0f Hz → %.1f Hz (f_max_utile = %.1f Hz)",
        D,
        f_s,
        f_s_new,
        f_max_utile,
    )

    iq_i = _scipy_decimate(iq.real.astype(np.float64), D, ftype="iir", zero_phase=True)
    iq_q = _scipy_decimate(iq.imag.astype(np.float64), D, ftype="iir", zero_phase=True)

    iq_decimated = (iq_i + 1j * iq_q).astype(np.complex64)

    logger.debug(
        "Décimation terminée — %d → %d échantillons",
        len(iq),
        len(iq_decimated),
    )
    return iq_decimated, f_s_new
