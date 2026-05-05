"""Respiration detection through band-power SNR thresholding."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate
from scipy.stats import f as _scipy_f

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Outcome of a single respiration-detection evaluation.

    Attributes
    ----------
    snr_db : float
        Measured SNR (dB) = 10·log10(P_signal / P_reference).
    alert : bool
        ``True`` if the Fisher p-value falls below the configured
        significance level *alpha* (binary decision).
    p_signal : float
        Mean power in the respiration band (linear).
    p_reference : float
        Mean power in the reference (noise) band (linear).
    p_value_f : float
        P-value of the Fisher F-test on the band-power ratio.
    acf_peak : float
        Normalised autocorrelation peak value within the respiration
        delay range (0 to 1, where 1 = perfect periodicity).
    fv_estimated : float or None
        Estimated breathing frequency (Hz) from the ACF peak delay,
        or ``None`` if the ACF could not be evaluated.
    score_presence : float
        Fused presence score in [0, 1], combining the spectral test
        (Fisher) and the time-domain test (ACF).
    """

    snr_db: float
    alert: bool
    p_signal: float
    p_reference: float
    p_value_f: float = 1.0
    acf_peak: float = 0.0
    fv_estimated: float | None = None
    score_presence: float = 0.0


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


# ======================================================================
# Single-column SNR for streaming mode
# ======================================================================

def detect_snr_column(
    col_db: np.ndarray,
    f_hz: np.ndarray,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
) -> float:
    """Compute the SNR of a single spectrum column.

    Parameters
    ----------
    col_db : numpy.ndarray
        Power spectrum in dB, shape ``(n_fft,)``.
    f_hz : numpy.ndarray
        Centred frequency axis (Hz), shape ``(n_fft,)``.
    bande_respiration : tuple[float, float]
        ``(f_lo, f_hi)`` — signal frequency band (Hz).
    bande_reference : tuple[float, float]
        ``(f_lo, f_hi)`` — reference (noise) band (Hz).

    Returns
    -------
    float
        SNR in dB for this single column.

    Raises
    ------
    ValueError
        If either frequency mask selects zero bins.

    Notes
    -----
    Streaming counterpart of :func:`snr_db_per_column`.  Operates on a
    single 1-D spectrum instead of a full 2-D spectrogram matrix,
    avoiding unnecessary array allocation in the real-time loop.
    """
    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference

    abs_f = np.abs(f_hz)
    mask_sig = (abs_f >= f_lo_sig) & (abs_f <= f_hi_sig)
    mask_ref = (abs_f >= f_lo_ref) & (abs_f <= f_hi_ref)

    if not np.any(mask_sig):
        raise ValueError(
            f"Masque bande respiration vide ({f_lo_sig}–{f_hi_sig} Hz). "
            f"Vérifier δf ou élargir la bande."
        )
    if not np.any(mask_ref):
        raise ValueError(
            f"Masque bande référence vide ({f_lo_ref}–{f_hi_ref} Hz). "
            f"Vérifier que la bande est dans la plage Nyquist."
        )

    col_lin = 10.0 ** (col_db / 10.0)
    p_sig = float(np.mean(col_lin[mask_sig]))
    p_ref = float(np.mean(col_lin[mask_ref]))

    eps = 1e-30
    return float(10.0 * np.log10(p_sig / (p_ref + eps)))


# ======================================================================
# Statistical detection — Fisher F-test, ACF, and fused score
# ======================================================================

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
        Linear-scale power spectrogram, shape ``(n_freq, n_time)`` or
        ``(n_freq,)`` for a single column.
    f_hz : numpy.ndarray
        Centred frequency axis (Hz), shape ``(n_freq,)``.
    bande_respiration : tuple[float, float]
        Signal frequency band (Hz) — same tuple as in :mod:`config.yaml`.
    bande_reference : tuple[float, float]
        Reference (noise) frequency band (Hz) — same tuple as in config.

    Returns
    -------
    tuple[float, float]
        ``(p_value, ratio)`` — F-test p-value and the band-power ratio
        ``mean(P_resp) / mean(P_ref)``.

    Raises
    ------
    ValueError
        If either frequency mask selects zero bins.

    Notes
    -----
    Under the null hypothesis H₀ of complex white Gaussian noise, the
    power of each FFT bin (after windowing) is proportional to a
    chi-squared variable with 2 degrees of freedom.  The mean of *n*
    such bins is therefore proportional to χ²(2n) / (2n), and the ratio
    of two independent means follows a Fisher distribution
    F(2·n_sig, 2·n_ref).  In practice we use F(n_sig, n_ref) as a
    conservative approximation that does not require multiplying the
    degrees of freedom.

    The returned p-value is the probability, under H₀, of observing a
    ratio at least as large as the one measured — i.e. directly the
    probability of false alarm of the test.
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
        Unwrapped instantaneous phase (1-D, radians).
    f_s : float
        Sampling rate of *phi* (Hz), i.e. the decimated rate.
    bande_respiration : tuple[float, float]
        ``(f_lo, f_hi)`` — same tuple as in config; the searched delay
        range is ``[1/f_hi, 1/f_lo]``.

    Returns
    -------
    tuple[float, float or None]
        ``(acf_peak, fv_estimated)``.  ``acf_peak`` is the normalised
        ACF value (in [-1, 1]) at the best lag, ``fv_estimated`` is
        ``1/tau_max`` (Hz) or ``None`` if no valid lag could be evaluated.

    Notes
    -----
    A periodic component at frequency *f_v* in the phase signal produces
    a strong peak in the autocorrelation at lag ``τ = 1/f_v``.  This is
    a time-domain test that is independent from (and therefore
    complementary to) the frequency-domain Fisher test.
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

    R_pos = R[n - 1:] / R0  # lags 0..n-1, normalised so R(0) = 1

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
        Normalised autocorrelation peak (assumed in [0, 1]; clipped
        otherwise).
    w : float, optional
        Weight of the Fisher-derived score (``score_F = 1 − p_value_f``).
        ``w = 1`` → purely spectral decision; ``w = 0`` → purely
        time-domain.  Default is 0.5.

    Returns
    -------
    float
        Fused presence score in [0, 1].
    """
    score_F = float(np.clip(1.0 - p_value_f, 0.0, 1.0))
    score_acf = float(np.clip(acf_peak, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    return float(w * score_F + (1.0 - w) * score_acf)


def detect_presence(
    S_db: np.ndarray,
    f_hz: np.ndarray,
    phi: np.ndarray,
    f_s: float,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
    alpha: float,
    w: float = 0.5,
) -> DetectionResult:
    """Detect respiratory presence using a fused Fisher / ACF test.

    Parameters
    ----------
    S_db : numpy.ndarray
        Power spectrogram in dB, shape ``(n_freq, n_time)``.
    f_hz : numpy.ndarray
        Centred frequency axis (Hz), shape ``(n_freq,)``.
    phi : numpy.ndarray
        Unwrapped instantaneous phase (1-D, radians) of the
        clutter-suppressed IQ signal at rate *f_s*.
    f_s : float
        Sampling rate of *phi* (Hz), i.e. the decimated rate.
    bande_respiration : tuple[float, float]
        Signal frequency band (Hz) — passed straight from config.
    bande_reference : tuple[float, float]
        Reference (noise) frequency band (Hz) — passed straight from
        config.
    alpha : float
        Significance level for the Fisher test (e.g. 0.01 = 1 % PFA).
        ``alert`` is ``True`` iff ``p_value_f < alpha``.
    w : float, optional
        Fusion weight (see :func:`_fusion_score`).  Default 0.5.

    Returns
    -------
    DetectionResult
        Full result dataclass with all statistical fields populated.

    Notes
    -----
    The binary alert is driven by the Fisher test (which has a clear
    statistical interpretation as a probability of false alarm), while
    *score_presence* drives the dashboard confidence curve.

    *snr_db* is preserved from the existing band-power computation so
    that legacy display code keeps working.
    """
    S_lin = 10.0 ** (S_db / 10.0)

    p_value_f, ratio = _fisher_pvalue(
        S_lin, f_hz, bande_respiration, bande_reference,
    )

    acf_peak, fv_estimated = _acf_peak(phi, f_s, bande_respiration)

    score_presence = _fusion_score(p_value_f, acf_peak, w=w)

    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference
    abs_f = np.abs(f_hz)
    mask_sig = (abs_f >= f_lo_sig) & (abs_f <= f_hi_sig)
    mask_ref = (abs_f >= f_lo_ref) & (abs_f <= f_hi_ref)
    p_signal = float(np.mean(S_lin[mask_sig]))
    p_reference = float(np.mean(S_lin[mask_ref]))
    eps = 1e-30
    snr_db = float(10.0 * np.log10(p_signal / (p_reference + eps)))

    alert = bool(p_value_f < alpha)

    logger.info(
        "Détection — p_F=%.2e (α=%.2e), ACF=%.2f @ fv=%s Hz, "
        "score=%.2f, SNR=%.1f dB → %s",
        p_value_f,
        alpha,
        acf_peak,
        f"{fv_estimated:.3f}" if fv_estimated is not None else "n/a",
        score_presence,
        snr_db,
        "ALERTE" if alert else "rien détecté",
    )

    return DetectionResult(
        snr_db=snr_db,
        alert=alert,
        p_signal=p_signal,
        p_reference=p_reference,
        p_value_f=p_value_f,
        acf_peak=acf_peak,
        fv_estimated=fv_estimated,
        score_presence=score_presence,
    )


def detect_presence_column(
    col_db: np.ndarray,
    f_hz: np.ndarray,
    phi_buffer: np.ndarray,
    f_s: float,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
    w: float = 0.5,
) -> tuple[float, float, float, float | None]:
    """Streaming counterpart of :func:`detect_presence` (single column).

    Parameters
    ----------
    col_db : numpy.ndarray
        Power spectrum (dB), 1-D — current STFT column.
    f_hz : numpy.ndarray
        Centred frequency axis (Hz).
    phi_buffer : numpy.ndarray
        Unwrapped phase segment corresponding to the same time window
        as *col_db* (1-D, radians).
    f_s : float
        Sampling rate of *phi_buffer* (Hz).
    bande_respiration : tuple[float, float]
        Signal frequency band (Hz) — from config.
    bande_reference : tuple[float, float]
        Reference frequency band (Hz) — from config.
    w : float, optional
        Fusion weight.  Default 0.5.

    Returns
    -------
    tuple[float, float, float, float or None]
        ``(score_presence, p_value_f, acf_peak, fv_estimated)``.
    """
    col_lin = 10.0 ** (col_db / 10.0)
    p_value_f, _ = _fisher_pvalue(
        col_lin, f_hz, bande_respiration, bande_reference,
    )
    acf_peak, fv_estimated = _acf_peak(phi_buffer, f_s, bande_respiration)
    score_presence = _fusion_score(p_value_f, acf_peak, w=w)
    return score_presence, p_value_f, acf_peak, fv_estimated
