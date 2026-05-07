"""Streaming respiration detection — Fisher F-test fused with phase ACF.

Batch (full-spectrogram) helpers are preserved in
:mod:`MicroDopplerDetection.legacy.detection_batch`.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import correlate
from scipy.stats import f as _scipy_f

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Internal building blocks (also re-used by the legacy batch wrappers)
# ----------------------------------------------------------------------

def _fisher_pvalue(
    S_lin: np.ndarray,
    f_hz: np.ndarray,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
) -> tuple[float, float]:
    """Fisher F-test on the band-power ratio.

    Parameters
    ----------
    S_lin : numpy.ndarray
        Linear-scale power spectrum (1-D column or 2-D spectrogram).
    f_hz : numpy.ndarray
        Centred frequency axis (Hz).
    bande_respiration, bande_reference : tuple[float, float]
        Signal and reference bands (Hz), as in ``config.yaml``.

    Returns
    -------
    tuple[float, float]
        ``(p_value, ratio)`` — F-test p-value and band-power ratio.

    Notes
    -----
    Under H₀ of complex white Gaussian noise, the FFT-bin powers are
    proportional to χ²(2) variables, so the ratio of band means follows
    F(n_sig, n_ref).  The p-value is the false-alarm probability of the
    observed ratio under H₀.
    """
    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference

    abs_f = np.abs(f_hz)
    mask_sig = (abs_f >= f_lo_sig) & (abs_f <= f_hi_sig)
    mask_ref = (abs_f >= f_lo_ref) & (abs_f <= f_hi_ref)

    n_sig = int(np.sum(mask_sig))
    n_ref = int(np.sum(mask_ref))

    if n_sig == 0:
        raise ValueError(
            f"Masque bande respiration vide ({f_lo_sig}–{f_hi_sig} Hz)."
        )
    if n_ref == 0:
        raise ValueError(
            f"Masque bande référence vide ({f_lo_ref}–{f_hi_ref} Hz)."
        )

    p_sig = float(np.mean(S_lin[mask_sig]))
    p_ref = float(np.mean(S_lin[mask_ref]))

    eps = 1e-30
    ratio = p_sig / (p_ref + eps)
    p_value = float(_scipy_f.sf(ratio, dfn=n_sig, dfd=n_ref))
    return p_value, ratio


def _acf_peak(
    phi: np.ndarray,
    f_s: float,
    bande_respiration: tuple[float, float],
) -> tuple[float, float | None]:
    """Normalised autocorrelation peak inside the respiration delay range.

    Parameters
    ----------
    phi : numpy.ndarray
        Unwrapped instantaneous phase (1-D, radians) of a clutter- and
        carrier-demodulated IQ segment.  **Length should be at least
        ``f_s / f_lo``** to capture one full breathing period.
    f_s : float
        Sampling rate of *phi* (Hz).
    bande_respiration : tuple[float, float]
        ``(f_lo, f_hi)`` — searched delay range is ``[1/f_hi, 1/f_lo]``.

    Returns
    -------
    tuple[float, float or None]
        ``(acf_peak, fv_estimated)`` — normalised ACF value at the best
        lag and corresponding ``1/τ_max`` estimate.
    """
    f_lo, f_hi = bande_respiration
    if f_lo <= 0 or f_hi <= 0 or f_hi <= f_lo:
        return 0.0, None

    phi = np.asarray(phi, dtype=np.float64)
    n = phi.size
    if n < 4:
        return 0.0, None

    phi_zm = phi - np.mean(phi)
    R = correlate(phi_zm, phi_zm, mode="full")
    R0 = R[n - 1]
    if R0 <= 0:
        return 0.0, None

    R_pos = R[n - 1:] / R0

    lag_min = int(np.ceil(f_s / f_hi))
    lag_max = int(np.floor(f_s / f_lo))
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, n - 1)

    if lag_max <= lag_min:
        return 0.0, None

    segment = R_pos[lag_min: lag_max + 1]
    idx_local = int(np.argmax(segment))
    tau_max = lag_min + idx_local
    acf_peak = float(segment[idx_local])
    fv_estimated = float(f_s / tau_max) if tau_max > 0 else None
    return acf_peak, fv_estimated


def _fusion_score(
    p_value_f: float,
    acf_peak: float,
    w: float = 0.5,
) -> float:
    """Fuse the Fisher and ACF scores into a presence score in [0, 1].

    Parameters
    ----------
    p_value_f : float
        P-value of the Fisher F-test.
    acf_peak : float
        Normalised autocorrelation peak (clipped to [0, 1]).
    w : float, optional
        Weight of the Fisher score.  ``w=1`` → purely spectral,
        ``w=0`` → purely time-domain.  Default 0.5.
    """
    score_F = float(np.clip(1.0 - p_value_f, 0.0, 1.0))
    score_acf = float(np.clip(acf_peak, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    return float(w * score_F + (1.0 - w) * score_acf)


# ----------------------------------------------------------------------
# Streaming entry point
# ----------------------------------------------------------------------

def detect_presence_column(
    col_db: np.ndarray,
    f_hz: np.ndarray,
    phi_buffer: np.ndarray,
    f_s: float,
    bande_respiration_spectral: tuple[float, float],
    bande_reference_spectral: tuple[float, float],
    bande_respiration_baseband: tuple[float, float],
    w: float = 0.5,
) -> tuple[float, float, float, float | None]:
    """Streaming detection on a single STFT column + matching phase segment.

    Parameters
    ----------
    col_db : numpy.ndarray
        Power spectrum (dB), 1-D — current STFT column (length n_fft).
    f_hz : numpy.ndarray
        Centred frequency axis (Hz), shape matches *col_db*.
    phi_buffer : numpy.ndarray
        Unwrapped phase of the **same time window** as *col_db*, after
        clutter suppression and (in cw_offset mode) carrier demodulation.
        Length should be ≥ ``f_s / bande_respiration_baseband[0]``.
    f_s : float
        Sampling rate of *phi_buffer* (Hz), i.e. the decimated rate.
    bande_respiration_spectral, bande_reference_spectral : tuple[float, float]
        Signal and reference bands **on the STFT axis** (Hz).  Already
        shifted by *f_offset* in CW-offset mode.
    bande_respiration_baseband : tuple[float, float]
        Respiration band **relative to the carrier** (Hz), e.g.
        ``(0.1, 0.8)``.  Used for the time-domain ACF test on the
        demodulated phase signal.
    w : float, optional
        Fusion weight (see :func:`_fusion_score`).  Default 0.5.

    Returns
    -------
    tuple[float, float, float, float or None]
        ``(score_presence, p_value_f, acf_peak, fv_estimated)``.
    """
    col_lin = 10.0 ** (col_db / 10.0)
    p_value_f, _ = _fisher_pvalue(
        col_lin, f_hz, bande_respiration_spectral, bande_reference_spectral,
    )

    acf_peak, fv_estimated = _acf_peak(
        phi_buffer, f_s, bande_respiration_baseband,
    )
    score_presence = _fusion_score(p_value_f, acf_peak, w=w)

    logger.debug(
        "Détection — p_F=%.2e, ACF=%.2f @ fv=%s Hz, score=%.2f",
        p_value_f,
        acf_peak,
        f"{fv_estimated:.3f}" if fv_estimated is not None else "n/a",
        score_presence,
    )
    return score_presence, p_value_f, acf_peak, fv_estimated
