"""Respiration detection through band-power SNR thresholding."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Outcome of a single respiration-detection evaluation.

    Attributes
    ----------
    snr_db : float
        Measured SNR (dB) = 10·log10(P_signal / P_reference).
    alert : bool
        ``True`` if *snr_db* exceeds the configured threshold.
    p_signal : float
        Mean power in the respiration band (linear).
    p_reference : float
        Mean power in the reference (noise) band (linear).
    """

    snr_db: float
    alert: bool
    p_signal: float
    p_reference: float


def detect_respiration(
    S_db: np.ndarray,
    f_hz: np.ndarray,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
    seuil_snr_dB: float,
) -> DetectionResult | None:
    """Evaluate the presence of a respiratory signature in the spectrogram.

    Parameters
    ----------
    S_db : numpy.ndarray
        Power spectrogram in dB, shape ``(n_freq, n_time)``.
    f_hz : numpy.ndarray
        Centred frequency axis (Hz), shape ``(n_freq,)``.
    bande_respiration : tuple[float, float]
        ``(f_lo, f_hi)`` — frequency band (Hz) where respiration energy is
        expected, e.g. ``(0.1, 0.8)``.
    bande_reference : tuple[float, float]
        ``(f_lo, f_hi)`` — reference band (Hz) for noise-floor estimation,
        e.g. ``(2.0, 5.0)``.
    seuil_snr_dB : float
        Detection threshold (dB).  An alert is raised when the measured
        SNR exceeds this value.

    Returns
    -------
    DetectionResult or None
        Detection outcome, or ``None`` if the frequency masks are empty
        (should not happen with correct config).

    Raises
    ------
    ValueError
        If the signal or reference frequency mask selects zero bins.
        This usually means the STFT frequency resolution is too coarse
        or the band limits are outside the Nyquist range.

    Notes
    -----
    The detector computes the **mean spectral power** in a narrow band
    around the expected breathing rate (0.1–0.8 Hz) and compares it to
    the power in a reference band well away from any respiratory harmonic.
    Because both bands undergo the same noise floor and processing gain,
    the ratio is a robust estimate of the micro-Doppler SNR.

    Both the positive (+f_v) and negative (−f_v) Doppler sidebands are
    included in the signal mask to capture energy from approaching and
    receding chest motion.
    """
    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference

    mask_sig = ((np.abs(f_hz) >= f_lo_sig) & (np.abs(f_hz) <= f_hi_sig))
    mask_ref = ((np.abs(f_hz) >= f_lo_ref) & (np.abs(f_hz) <= f_hi_ref))

    if not np.any(mask_sig):
        raise ValueError(
            f"Masque de la bande respiration vide ({f_lo_sig}–{f_hi_sig} Hz). "
            f"Vérifier la résolution fréquentielle (δf) ou élargir la bande."
        )
    if not np.any(mask_ref):
        raise ValueError(
            f"Masque de la bande de référence vide ({f_lo_ref}–{f_hi_ref} Hz). "
            f"Vérifier que f_max_utile couvre cette bande après décimation."
        )

    S_lin = 10.0 ** (S_db / 10.0)

    p_signal = float(np.mean(S_lin[mask_sig, :]))
    p_reference = float(np.mean(S_lin[mask_ref, :]))

    eps = 1e-30
    snr_db = 10.0 * np.log10(p_signal / (p_reference + eps))
    alert = bool(snr_db >= seuil_snr_dB)

    logger.info(
        "Détection — SNR = %.1f dB (seuil = %.1f dB) → %s",
        snr_db,
        seuil_snr_dB,
        "ALERTE" if alert else "rien détecté",
    )

    return DetectionResult(
        snr_db=snr_db,
        alert=alert,
        p_signal=p_signal,
        p_reference=p_reference,
    )


def snr_db_per_column(
    S_db: np.ndarray,
    f_hz: np.ndarray,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame SNR for the dashboard confidence curve.

    Parameters
    ----------
    S_db : numpy.ndarray
        Power spectrogram in dB, shape ``(n_freq, n_time)``.
    f_hz : numpy.ndarray
        Centred frequency axis (Hz), shape ``(n_freq,)``.
    bande_respiration : tuple[float, float]
        Signal frequency band (Hz).
    bande_reference : tuple[float, float]
        Reference (noise) frequency band (Hz).

    Returns
    -------
    snr : numpy.ndarray
        SNR per STFT column (dB), shape ``(n_time,)``.
    p_sig : numpy.ndarray
        Signal-band mean power per column (linear), shape ``(n_time,)``.
    p_ref : numpy.ndarray
        Reference-band mean power per column (linear), shape ``(n_time,)``.

    Raises
    ------
    ValueError
        If either frequency mask is empty.

    Notes
    -----
    This function provides a **time-resolved** view of the detection
    confidence.  Each STFT column (one slow-time frame) yields one SNR
    sample.  Plotting this series allows the operator to see when
    breathing starts or stops and to assess the temporal stability of the
    detection.
    """
    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference

    mask_sig = ((np.abs(f_hz) >= f_lo_sig) & (np.abs(f_hz) <= f_hi_sig))
    mask_ref = ((np.abs(f_hz) >= f_lo_ref) & (np.abs(f_hz) <= f_hi_ref))

    if not np.any(mask_sig):
        raise ValueError(
            f"Masque bande respiration vide ({f_lo_sig}–{f_hi_sig} Hz)."
        )
    if not np.any(mask_ref):
        raise ValueError(
            f"Masque bande référence vide ({f_lo_ref}–{f_hi_ref} Hz)."
        )

    S_lin = 10.0 ** (S_db / 10.0)

    p_sig = np.mean(S_lin[mask_sig, :], axis=0)
    p_ref = np.mean(S_lin[mask_ref, :], axis=0)

    eps = 1e-30
    snr = 10.0 * np.log10(p_sig / (p_ref + eps))

    return snr, p_sig, p_ref
