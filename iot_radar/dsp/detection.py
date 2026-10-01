"""Breathing detection of the micro-Doppler chain: Fisher F-test fused with a phase ACF.

Place in the chain: last processing step, run on every STFT column.  Two
indicators are combined into a presence score in [0, 1]:

* a **spectral** test — is there more power in the breathing band (around the
  carrier) than in a noise-only reference band?  (Fisher F-test,
  :func:`_fisher_p_value`);
* a **temporal** test — is the demodulated phase periodic at a breathing
  rate?  (normalised autocorrelation peak, :func:`_acf_peak`).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import correlate
from scipy.stats import f as f_distribution

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------

def _fisher_p_value(
    power_lin: np.ndarray,
    f_hz: np.ndarray,
    breathing_band_hz: tuple[float, float],
    reference_band_hz: tuple[float, float],
    f_center_hz: float = 0.0,
) -> tuple[float, float]:
    """Fisher F-test on the ratio of the mean powers of two bands.

    Parameters
    ----------
    power_lin : numpy.ndarray
        Power spectrum in linear scale, shape ``(n_fft,)``.
    f_hz : numpy.ndarray
        Matching centred frequency axis (Hz).
    breathing_band_hz, reference_band_hz : tuple[float, float]
        Signal and reference bands ``(low, high)`` in Hz, **relative to the
        carrier** (baseband offsets), as in ``configs/radar.yaml``.
    f_center_hz : float, optional
        Position of the carrier in the spectrum (Hz).  Both bands are taken
        on the two sidebands around it, i.e. on ``|f_hz - f_center_hz|``:
        the TX offset in CW-offset mode, 0 in pure CW.

    Returns
    -------
    tuple[float, float]
        ``(p_value, ratio)`` — p-value of the F-test and ratio of the mean
        band powers.

    Notes
    -----
    Under H0 (complex white Gaussian noise), the power of each FFT bin is
    proportional to a χ²(2) variable, so the ratio of the band means follows
    F(2·n_signal, 2·n_reference).  Adjacent bins of a windowed FFT are in
    fact correlated, which makes the p-values too small (more false alarms).
    """
    signal_low_hz, signal_high_hz = breathing_band_hz
    reference_low_hz, reference_high_hz = reference_band_hz

    offset_hz = np.abs(f_hz - f_center_hz)
    signal_mask = (offset_hz >= signal_low_hz) & (offset_hz <= signal_high_hz)
    reference_mask = (offset_hz >= reference_low_hz) & (offset_hz <= reference_high_hz)

    n_signal = int(np.sum(signal_mask))
    n_reference = int(np.sum(reference_mask))

    if n_signal == 0:
        raise ValueError(
            f"Empty breathing band ({signal_low_hz}–{signal_high_hz} Hz)."
        )
    if n_reference == 0:
        raise ValueError(
            f"Empty reference band ({reference_low_hz}–{reference_high_hz} Hz)."
        )

    signal_power = float(np.mean(power_lin[signal_mask]))
    reference_power = float(np.mean(power_lin[reference_mask]))

    eps = 1e-30
    ratio = signal_power / (reference_power + eps)
    p_value = float(f_distribution.sf(ratio, dfn=2*n_signal, dfd=2*n_reference))
    return p_value, ratio


def _acf_peak(
    phase_rad: np.ndarray,
    f_s_hz: float,
    breathing_band_hz: tuple[float, float],
) -> tuple[float, float | None]:
    """Normalised autocorrelation peak inside the breathing delay range.

    Parameters
    ----------
    phase_rad : numpy.ndarray
        Unwrapped instantaneous phase (rad, 1-D) of the clutter-filtered,
        carrier-demodulated IQ.  Its length should be at least
        ``f_s_hz / breathing_band_hz[0]`` to hold one full breathing period.
    f_s_hz : float
        Sampling rate of *phase_rad* (Hz).
    breathing_band_hz : tuple[float, float]
        ``(low, high)``: the searched delays are ``[1/high, 1/low]``.

    Returns
    -------
    tuple[float, float or None]
        ``(acf_peak, breathing_rate_hz)`` — unbiased normalised ACF at the
        best delay ``τ`` and the matching rate ``1 / τ`` (``None`` when the
        estimate is impossible).
    """
    f_low_hz, f_high_hz = breathing_band_hz
    if f_low_hz <= 0 or f_high_hz <= 0 or f_high_hz <= f_low_hz:
        return 0.0, None

    phase_rad = np.asarray(phase_rad, dtype=np.float64)
    n = phase_rad.size
    if n < 4:
        return 0.0, None

    centred = phase_rad - np.mean(phase_rad)
    acf = correlate(centred, centred, mode="full")
    acf_at_zero = acf[n - 1]
    if acf_at_zero <= 0:
        return 0.0, None

    acf_positive_lags = acf[n - 1:] / acf_at_zero

    lag_min = int(np.ceil(f_s_hz / f_high_hz))
    lag_max = int(np.floor(f_s_hz / f_low_hz))
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, n - 1)

    if lag_max <= lag_min:
        return 0.0, None

    # Unbiased estimate: compensate the decreasing number of terms per lag.
    lags = np.arange(lag_min, lag_max + 1)
    searched = acf_positive_lags[lag_min: lag_max + 1] * (n / (n - lags))
    best_index = int(np.argmax(searched))
    best_lag = lag_min + best_index
    acf_peak = float(searched[best_index])
    breathing_rate_hz = float(f_s_hz / best_lag) if best_lag > 0 else None
    return acf_peak, breathing_rate_hz


def _fusion_score(
    p_value: float,
    acf_peak: float,
    spectral_weight: float,
    p_value_decades: float,
    acf_floor: float,
    acf_good: float,
) -> float:
    """Fuse the Fisher and ACF indicators into a presence score in [0, 1].

    Each raw indicator is first mapped to ``[0, 1]``:

    * spectral — the F-test p-value on a log scale: a p-value of
      ``10**(-p_value_decades)`` (or smaller) gives 1 and a p-value of 1
      gives 0 (confidence grows by orders of magnitude);
    * temporal — the ACF peak, ramped linearly between a noise floor
      (``acf_floor`` → 0) and a clear periodicity (``acf_good`` → 1).

    The score is ``spectral_weight · spectral + (1 − spectral_weight) · temporal``.

    Parameters
    ----------
    p_value : float
        p-value of the Fisher F-test.
    acf_peak : float
        Normalised autocorrelation peak (may be negative).
    spectral_weight : float
        Weight of the spectral score, in [0, 1].
    p_value_decades : float
        Number of decades below 1 at which the spectral score saturates.
    acf_floor, acf_good : float
        ACF peak values mapped to 0 and 1.
    """
    if p_value_decades <= 0:
        raise ValueError("p_value_decades must be > 0.")
    if acf_good <= acf_floor:
        raise ValueError("acf_good must be > acf_floor.")
    if spectral_weight < 0 or spectral_weight > 1:
        raise ValueError("spectral_weight must be between 0 and 1.")

    eps = 1e-30
    spectral_score = float(
        np.clip(-np.log10(p_value + eps) / p_value_decades, 0.0, 1.0)
    )
    temporal_score = float(
        np.clip((acf_peak - acf_floor) / (acf_good - acf_floor), 0.0, 1.0)
    )
    return float(spectral_weight * spectral_score + (1.0 - spectral_weight) * temporal_score)


# ----------------------------------------------------------------------
# Streaming entry point
# ----------------------------------------------------------------------

def detect_presence_column(
    column_db: np.ndarray,
    f_hz: np.ndarray,
    phase_rad: np.ndarray,
    f_s_hz: float,
    breathing_band_hz: tuple[float, float],
    reference_band_hz: tuple[float, float],
    f_center_hz: float,
    spectral_weight: float,
    p_value_decades: float,
    acf_floor: float,
    acf_good: float,
) -> tuple[float, float, float, float | None]:
    """Breathing detection on one STFT column and the recent phase history.

    Parameters
    ----------
    column_db : numpy.ndarray
        Power spectrum (dB) of the current STFT column, shape ``(n_fft,)``.
    f_hz : numpy.ndarray
        Matching centred frequency axis (Hz).
    phase_rad : numpy.ndarray
        Unwrapped phase (rad) of the clutter-filtered, carrier-demodulated
        IQ over the last ``detection.acf_buffer_s`` seconds (longer than the
        STFT segment).  Its length should be ≥ ``f_s_hz / breathing_band_hz[0]``.
    f_s_hz : float
        Sampling rate of *phase_rad* (Hz), i.e. the decimated rate.
    breathing_band_hz, reference_band_hz : tuple[float, float]
        Bands **relative to the carrier** (Hz), e.g. ``(0.1, 0.8)`` and
        ``(2.0, 5.0)``; the breathing band also sets the delays searched by
        the ACF.
    f_center_hz : float
        Position of the carrier in the spectrum (Hz): the TX offset in
        CW-offset mode, 0 in pure CW.
    spectral_weight, p_value_decades, acf_floor, acf_good : float
        Score-mapping parameters, see :func:`_fusion_score`.

    Returns
    -------
    tuple[float, float, float, float or None]
        ``(presence_score, p_value, acf_peak, breathing_rate_hz)``.
    """
    column_lin = 10.0 ** (column_db / 10.0)
    p_value, _ = _fisher_p_value(
        column_lin, f_hz, breathing_band_hz, reference_band_hz, f_center_hz=f_center_hz,
    )

    acf_peak, breathing_rate_hz = _acf_peak(
        phase_rad, f_s_hz, breathing_band_hz,
    )
    presence_score = _fusion_score(
        p_value,
        acf_peak,
        spectral_weight=spectral_weight,
        p_value_decades=p_value_decades,
        acf_floor=acf_floor,
        acf_good=acf_good,
    )

    logger.debug(
        "Detection — p=%.2e, ACF=%.2f @ %s Hz, score=%.2f",
        p_value,
        acf_peak,
        f"{breathing_rate_hz:.3f}" if breathing_rate_hz is not None else "n/a",
        presence_score,
    )
    return presence_score, p_value, acf_peak, breathing_rate_hz
