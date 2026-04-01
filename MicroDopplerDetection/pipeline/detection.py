"""
Energy-based detection in selected Doppler bands (respiration vs reference).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DetectionMode = Literal["threshold", "none"]


@dataclass
class DetectionResult:
    """Outputs of the threshold detector."""

    snr_db: float
    """Average in-band signal power vs reference band (dB)."""

    alert: bool
    """True if ``snr_db`` exceeds the configured threshold."""

    p_signal: float
    """Mean linear power in the respiration mask."""

    p_reference: float
    """Mean linear power in the reference (noise) mask."""


def _band_mask(f_hz: np.ndarray, f_lo: float, f_hi: float, use_abs: bool = True) -> np.ndarray:
    if use_abs:
        fa = np.abs(f_hz)
        return (fa >= f_lo) & (fa <= f_hi)
    return (f_hz >= f_lo) & (f_hz <= f_hi)


def detect_respiration(
    Z: np.ndarray,
    f_hz: np.ndarray,
    band_respiration_hz: tuple[float, float],
    band_reference_hz: tuple[float, float],
    seuil_snr_db: float,
    mode: DetectionMode = "threshold",
    df_hz: float | None = None,
) -> DetectionResult | None:
    """
    Compare average energy near respiration Doppler to a higher reference band.

    Parameters
    ----------
    Z
        Complex STFT matrix, shape ``(n_fft, n_time)`` (fft-shifted along freq).
    f_hz
        Signed Doppler frequency axis (Hz), shape ``(n_fft,)``, consistent with
        fft-shift applied to ``Z``.
    band_respiration_hz
        Tuple ``(f_lo, f_hi)`` in Hz for the search band (typical ``(0.1, 0.8)``).
        Magnitudes ``|f|`` are used so both ``+fv`` and ``-fv`` lines contribute.
    band_reference_hz
        Tuple ``(f_lo, f_hi)`` for noise floor (typical ``(2.0, 5.0)``).
    seuil_snr_db
        Detection threshold on SNR (dB) (typical ``10``).
    mode
        ``"threshold"`` runs the test; ``"none"`` skips and returns ``None``.
    df_hz
        STFT bin width (Hz). If provided, frequency masks are widened by
        ``±df_hz/2`` so narrow bands still intersect FFT bins after decimation.

    Returns
    -------
    DetectionResult or None
        Metrics when ``mode=="threshold"``, else ``None``.

    Notes
    -----
    Physical note: after clutter suppression, residual wideband noise and
    phase noise elevate distant Doppler bins; a line at breathing rate stands
    out locally. Comparing a low-frequency band to a higher reference band
    approximates an SNR without requiring a parametric spectral estimator.
    """
    if mode == "none":
        logger.info("Detection disabled (mode=none).")
        return None

    if mode != "threshold":
        raise ValueError(f"Unknown detection mode: {mode!r}")

    Z = np.asarray(Z, dtype=np.complex64)
    f_hz = np.asarray(f_hz, dtype=np.float64)

    r_lo, r_hi = float(band_respiration_hz[0]), float(band_respiration_hz[1])
    n_lo, n_hi = float(band_reference_hz[0]), float(band_reference_hz[1])

    half = 0.0
    if df_hz is not None and df_hz > 0:
        half = float(df_hz) / 2.0

    mask_s = _band_mask(f_hz, max(0.0, r_lo - half), r_hi + half, use_abs=True)
    mask_n = _band_mask(f_hz, max(0.0, n_lo - half), n_hi + half, use_abs=True)

    if not np.any(mask_s) or not np.any(mask_n):
        raise ValueError(
            "Empty frequency mask after widening; increase spectrogramme.n_fft "
            "or adjust detection bands vs decimated sample rate."
        )

    power = np.mean(np.abs(Z[mask_s, :]) ** 2, axis=0)
    p_signal = float(np.mean(power))

    p_noise = np.mean(np.abs(Z[mask_n, :]) ** 2, axis=0)
    p_reference = float(np.mean(p_noise))

    floor = 1e-18
    snr_db = 10.0 * np.log10((p_signal + floor) / (p_reference + floor))
    alert = bool(snr_db > float(seuil_snr_db))

    logger.info(
        "Detection: P_sig=%.4g P_ref=%.4g SNR=%.2f dB threshold=%.2f dB -> %s",
        p_signal,
        p_reference,
        snr_db,
        seuil_snr_db,
        "ALERT" if alert else "no alert",
    )

    return DetectionResult(snr_db=float(snr_db), alert=alert, p_signal=p_signal, p_reference=p_reference)
