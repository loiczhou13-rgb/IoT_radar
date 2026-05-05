"""Sample-rate reduction through cascaded low-pass filtering and down-sampling."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import decimate as _scipy_decimate

logger = logging.getLogger(__name__)

_MAX_SINGLE_STAGE = 13
_PRIMES = (2, 3, 5, 7, 11, 13)


def _factorise(D: int) -> list[int]:
    """Decompose *D* into prime factors ≤ 13.

    Each factor becomes one decimation stage with its own Chebyshev
    anti-aliasing filter, maximising stopband rejection per stage.

    Raises
    ------
    ValueError
        If *D* has a prime factor > 13.
    """
    if D <= 1:
        return []

    factors: list[int] = []
    remaining = D

    for p in _PRIMES:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p

    if remaining > 1:
        raise ValueError(
            f"D={D} contient un facteur premier > 13 (résidu={remaining}). "
            f"Choisir un D décomposable en petits facteurs ({', '.join(map(str, _PRIMES))})."
        )

    factors.sort()
    return factors


def _decimate_1d(x: np.ndarray, q: int) -> np.ndarray:
    """Decimate a real 1-D array by factor *q* (single stage)."""
    return _scipy_decimate(x, q, ftype="iir", zero_phase=True)


def decimate_iq(
    iq: np.ndarray,
    f_s: float,
    D: int,
    f_max_utile: float,
) -> tuple[np.ndarray, float]:
    """Decimate the IQ vector by factor *D* using cascaded stages.

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ samples at rate *f_s*.
    f_s : float
        Current sampling rate (Hz), e.g. 2.0e6.
    D : int
        Total decimation factor.  Must be decomposable into prime
        factors ≤ 13.  The output rate is ``f_s / D``.
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
        If the Shannon–Nyquist criterion (with filter transition-band
        margin) is violated after decimation, i.e.
        ``f_s / D <= 2.5 * f_max_utile``, or if *D* contains a prime
        factor > 13.

    Notes
    -----
    *D* is decomposed into its prime factors and each factor becomes a
    separate decimation stage.  This follows the ``scipy.signal.decimate``
    recommendation to keep each stage's factor ≤ 13 for adequate
    anti-aliasing with the built-in Chebyshev type-I order-8 filter
    (≈ −33 dB rejection per stage of 2, −24 dB per stage of 5).

    I and Q channels are decimated independently to preserve the analytic
    (complex) nature of the signal.
    """
    f_s_new = f_s / D

    if f_s_new <= 2.5 * f_max_utile:
        raise ValueError(
            f"Critère de Shannon (avec marge filtre anti-repliement) violé : "
            f"f_s_new = {f_s_new:.1f} Hz ≤ 2.5 × f_max_utile = {2.5 * f_max_utile:.1f} Hz. "
            f"Réduire D (actuellement {D}) ou augmenter f_s."
        )

    stages = _factorise(D)

    logger.debug(
        "Décimation ×%d en %d étage(s) %s — f_s : %.0f → %.1f Hz",
        D,
        len(stages),
        stages,
        f_s,
        f_s_new,
    )

    iq_i = iq.real.astype(np.float64)
    iq_q = iq.imag.astype(np.float64)

    f_cur = f_s
    for i, q in enumerate(stages):
        f_cut = 0.8 * (f_cur / q) / 2.0
        logger.debug(
            "  Étage %d/%d : ×%d — f_s=%.1f Hz → %.1f Hz, f_coupure≈%.1f Hz",
            i + 1,
            len(stages),
            q,
            f_cur,
            f_cur / q,
            f_cut,
        )
        iq_i = _decimate_1d(iq_i, q)
        iq_q = _decimate_1d(iq_q, q)
        f_cur /= q

    iq_decimated = (iq_i + 1j * iq_q).astype(np.complex64)

    logger.debug(
        "Décimation terminée — %d → %d échantillons",
        len(iq),
        len(iq_decimated),
    )
    return iq_decimated, f_s_new
