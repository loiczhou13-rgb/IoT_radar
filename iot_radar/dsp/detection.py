"""Breathing detection.

Place in the chain: last processing step.  Two detectors live here until the
micro-Doppler chain is removed:

**Phase chain (current)** — :func:`breathing_metrics` and
:class:`BreathingDetector`.  The logic mirrors commercial radar vital-sign
pipelines (TI mmWave "vital signs" demo, Acconeer / Infineon breathing
applications): per analysis window, a few **spectral quality metrics** of the
chest displacement give an instantaneous confidence; a **temporal decision
layer** smooths it and applies hysteresis so that the state does not flicker.

* Breathing rate — highest peak of the displacement spectrum in the
  breathing band (zero-padded, parabolic interpolation).
* Genuine peak — the maximum must be a local maximum of the spectrum; a
  maximum on a band edge while the spectrum keeps rising outside the band is
  the skirt of an out-of-band component, not a breathing rate.
* Band SNR — peak power / noise floor (median power in a reference band
  free of vital signs, e.g. 2.5–5 Hz).
* Peak concentration — share of the breathing-band energy within
  ±``peak_halfwidth_hz`` of the peak (TI's "confidence metric").
* Motion — body movements give displacements far larger than breathing;
  above ``motion_ptp_mm`` peak to peak the window is declared corrupted.

The instantaneous confidence is the geometric mean of the SNR and
concentration scores, forced to 0 during motion or without a genuine peak.
:class:`BreathingDetector` smooths it exponentially, applies the ON / OFF
thresholds, holds the ``MOTION`` state, reports the median rate and raises
an apnea alert when no inhalation has been seen for ``apnea_s`` seconds
although breathing was detected shortly before (a radar cannot tell a
breath-hold from the person leaving the beam).

**Micro-Doppler chain (former)** — a Fisher F-test on one STFT column fused
with the autocorrelation peak of the phase (:func:`detect_presence_column`).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.signal import correlate
from scipy.stats import f as f_distribution

from iot_radar.dsp.estimation import Spectrum

logger = logging.getLogger(__name__)

STATE_WARMUP = "WARMUP"
STATE_NO_BREATHING = "NO_BREATHING"
STATE_BREATHING = "BREATHING"
STATE_MOTION = "MOTION"
STATE_CODES = {STATE_WARMUP: -1, STATE_NO_BREATHING: 0, STATE_BREATHING: 1, STATE_MOTION: 2}
"""Integer code of each detector state (for arrays and plots)."""


# ----------------------------------------------------------------------
# Phase chain: per-window metrics
# ----------------------------------------------------------------------

@dataclass
class DetectionSettings:
    """Parameters of the breathing detector (``vital_signs.detection`` section).

    Attributes
    ----------
    snr_min_db, snr_good_db : float
        Band SNR mapped to the scores 0 and 1 (dB).
    peak_halfwidth_hz : float
        Half-width of the window around the peak for the concentration (Hz).
    concentration_floor, concentration_good : float
        Peak concentration mapped to the scores 0 and 1.
    motion_ptp_mm : float
        Peak-to-peak displacement above which a window is "motion" (mm).
    motion_window_s : float
        Length of the recent segment also checked for motion (s).
    motion_hold_s : float
        Time the ``MOTION`` state is held after the last corrupted window (s).
    smoothing_tau_s : float
        Time constant of the exponential smoothing of the confidence (s).
    threshold_on, threshold_off : float
        Smoothed confidence entering / leaving ``BREATHING`` (hysteresis).
    rate_median_len : int
        Reported rate = median of the last *rate_median_len* reliable rates.
    apnea_s : float
        Time without inhalation that raises the apnea alert (s).
    apnea_memory_s : float
        The alert is raised only if breathing was detected less than this ago (s).
    """

    snr_min_db: float = 8.0
    snr_good_db: float = 16.0
    peak_halfwidth_hz: float = 0.1
    concentration_floor: float = 0.55
    concentration_good: float = 0.85
    motion_ptp_mm: float = 30.0
    motion_window_s: float = 5.0
    motion_hold_s: float = 3.0
    smoothing_tau_s: float = 4.0
    threshold_on: float = 0.6
    threshold_off: float = 0.4
    rate_median_len: int = 5
    apnea_s: float = 10.0
    apnea_memory_s: float = 30.0

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "DetectionSettings":
        """Build from a YAML mapping; missing keys keep their default."""
        known = {f.name: section[f.name] for f in fields(cls) if f.name in section}
        settings = cls(**known)
        if not settings.threshold_off <= settings.threshold_on:
            raise ValueError("threshold_off must be <= threshold_on.")
        if (settings.snr_good_db <= settings.snr_min_db
                or settings.concentration_good <= settings.concentration_floor):
            raise ValueError("The 'good' values must exceed the corresponding floors.")
        return settings


@dataclass
class BreathMetrics:
    """Spectral quality metrics of one analysis window.

    Attributes
    ----------
    rate_hz : float or None
        Spectral-peak breathing rate (Hz).
    genuine_peak : bool
        ``False`` when the in-band maximum is only the skirt of an
        out-of-band component.
    snr_db : float
        Peak power over the reference-band noise floor (dB).
    noise_floor : float
        Median power in the reference band (linear, unit of the spectrum).
    concentration : float
        Share of the breathing-band energy within ±``peak_halfwidth_hz`` of the peak.
    motion : bool
        Large body movement detected (window unusable).
    displacement_ptp_mm : float
        Peak-to-peak displacement used for the motion test (mm; NaN if unknown).
    snr_score, concentration_score, confidence : float
        Scores in [0, 1]; ``confidence`` is the instantaneous confidence.
    """

    rate_hz: float | None
    genuine_peak: bool
    snr_db: float
    noise_floor: float
    concentration: float
    motion: bool
    displacement_ptp_mm: float
    snr_score: float
    concentration_score: float
    confidence: float


def ramp(value: float, floor: float, good: float) -> float:
    """Linear map: ``value ≤ floor`` → 0, ``value ≥ good`` → 1 (NaN → 0)."""
    if not np.isfinite(value):
        return 0.0
    return float(np.clip((value - floor) / (good - floor), 0.0, 1.0))


def _parabolic_peak_hz(f_hz: np.ndarray, psd: np.ndarray, k: int) -> float:
    """Frequency of the peak at bin *k*, refined by a parabola through the log-power."""
    if 0 < k < psd.size - 1:
        left, centre, right = np.log(psd[k - 1: k + 2] + 1e-300)
        curvature = left - 2.0 * centre + right
        offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)) if curvature != 0 else 0.0
        return float(f_hz[k] + offset * (f_hz[1] - f_hz[0]))
    return float(f_hz[k])


def breathing_metrics(
    spectrum: Spectrum,
    displacement_ptp_mm: float,
    breathing_band_hz: tuple[float, float],
    reference_band_hz: tuple[float, float],
    settings: DetectionSettings,
) -> BreathMetrics:
    """Per-window metrics and instantaneous confidence.

    Parameters
    ----------
    spectrum : Spectrum
        Periodogram of the detrended (not band-passed) displacement.
    displacement_ptp_mm : float
        Peak-to-peak displacement (mm), or NaN when the demodulation is not
        calibrated (linear fallback): motion is then not assessed.
    breathing_band_hz, reference_band_hz : tuple[float, float]
        Bands (Hz).
    settings : DetectionSettings
        Detector parameters.
    """
    f_hz, psd = spectrum.f_hz, spectrum.psd
    in_band = (f_hz >= breathing_band_hz[0]) & (f_hz <= breathing_band_hz[1])
    in_reference = (f_hz >= reference_band_hz[0]) & (f_hz <= reference_band_hz[1])
    if not in_band.any() or not in_reference.any():
        raise ValueError("Breathing or reference band outside the spectrum.")

    band_indices = np.flatnonzero(in_band)
    k = int(band_indices[np.argmax(psd[band_indices])])
    rises_to_the_left = k > 0 and psd[k - 1] > psd[k]
    rises_to_the_right = k < psd.size - 1 and psd[k + 1] > psd[k]
    genuine = not (rises_to_the_left or rises_to_the_right)
    rate_hz = _parabolic_peak_hz(f_hz, psd, k)
    noise_floor = float(max(np.median(psd[in_reference]), 1e-300))
    snr_db = float(10.0 * np.log10(psd[k] / noise_floor))

    around_peak = in_band & (np.abs(f_hz - f_hz[k]) <= settings.peak_halfwidth_hz)
    band_energy = float(psd[in_band].sum())
    concentration = float(psd[around_peak].sum() / band_energy) if band_energy > 0 else 0.0

    motion = bool(np.isfinite(displacement_ptp_mm) and displacement_ptp_mm > settings.motion_ptp_mm)
    snr_score = ramp(snr_db, settings.snr_min_db, settings.snr_good_db)
    concentration_score = ramp(concentration, settings.concentration_floor, settings.concentration_good)
    if motion or not genuine:
        confidence = 0.0
    else:
        confidence = math.sqrt(snr_score * concentration_score)
    return BreathMetrics(
        rate_hz=rate_hz,
        genuine_peak=genuine,
        snr_db=snr_db,
        noise_floor=noise_floor,
        concentration=concentration,
        motion=motion,
        displacement_ptp_mm=float(displacement_ptp_mm),
        snr_score=snr_score,
        concentration_score=concentration_score,
        confidence=confidence,
    )


# ----------------------------------------------------------------------
# Phase chain: temporal decision
# ----------------------------------------------------------------------

@dataclass
class BreathingState:
    """Output of :class:`BreathingDetector` for one analysis update.

    Attributes
    ----------
    t_s : float
        Time of the update (s, stream time).
    state : str
        ``WARMUP``, ``NO_BREATHING``, ``BREATHING`` or ``MOTION``.
    confidence : float
        Smoothed confidence in [0, 1] (drives the decision).
    confidence_instant : float
        Instantaneous confidence of the current window.
    rate_bpm : float or None
        Median breathing rate (breaths / min), ``None`` if not breathing.
    detected : bool
        ``True`` in the ``BREATHING`` state.
    metrics : BreathMetrics or None
        Metrics of the current window.
    breathing_for_s : float
        Duration of the current ``BREATHING`` episode (s, 0 otherwise).
    since_last_breath_s : float
        Time since the last detected inhalation (s, NaN if unknown).
    apnea : bool
        Apnea (or target lost) suspected.
    """

    t_s: float
    state: str
    confidence: float
    confidence_instant: float
    rate_bpm: float | None
    detected: bool
    metrics: BreathMetrics | None
    breathing_for_s: float = 0.0
    since_last_breath_s: float = float("nan")
    apnea: bool = False


class BreathingDetector:
    """Temporal decision layer: smoothing, hysteresis, motion hold, apnea alert.

    One instance per stream; call :meth:`update` once per analysis window.

    Parameters
    ----------
    settings : DetectionSettings
        Detector parameters.
    period_s : float
        Nominal time between two updates, used for the very first smoothing
        step (the next ones use the actual time differences).

    Notes
    -----
    The smoothed confidence starts at 0, so a detection needs about
    ``−τ·ln(1 − threshold_on)`` seconds of consistent evidence (≈ 3.7 s with
    the defaults) — a protection against isolated false alarms.
    """

    def __init__(self, settings: DetectionSettings, period_s: float) -> None:
        self.settings = settings
        self.period_s = float(period_s)
        self._confidence = 0.0
        self._breathing = False
        self._t_last_s: float | None = None
        self._t_motion_s: float | None = None
        self._t_breathing_start_s: float | None = None
        self._t_last_breathing_s: float | None = None
        self._rates_bpm: deque[float] = deque(maxlen=max(1, int(settings.rate_median_len)))

    def warmup(self, t_s: float) -> BreathingState:
        """State reported while the analysis window is still filling."""
        return BreathingState(t_s, STATE_WARMUP, 0.0, 0.0, None, False, None)

    def update(
        self,
        metrics: BreathMetrics,
        t_s: float,
        since_last_breath_s: float = float("nan"),
    ) -> BreathingState:
        """Fold the metrics of one window into the decision; return the new state.

        *since_last_breath_s* (from :func:`iot_radar.dsp.estimation.breath_cycles`)
        drives the apnea alert.
        """
        s = self.settings
        if self._t_last_s is None:
            dt_s = self.period_s
        else:
            dt_s = max(0.0, t_s - self._t_last_s)
        self._t_last_s = t_s

        if metrics.motion:
            self._t_motion_s = t_s
        in_motion = self._t_motion_s is not None and t_s - self._t_motion_s <= s.motion_hold_s
        target = 0.0 if in_motion else metrics.confidence
        alpha = 1.0 - math.exp(-dt_s / s.smoothing_tau_s) if s.smoothing_tau_s > 0 else 1.0
        self._confidence += alpha * (target - self._confidence)

        if in_motion:
            # Motion invalidates the measurement: keep no rate history.
            self._breathing = False
            self._t_breathing_start_s = None
            self._rates_bpm.clear()
            return BreathingState(t_s, STATE_MOTION, self._confidence, target, None, False, metrics,
                                  since_last_breath_s=since_last_breath_s)

        if metrics.confidence >= s.threshold_off and metrics.rate_hz is not None:
            self._rates_bpm.append(60.0 * metrics.rate_hz)
        if self._breathing and self._confidence < s.threshold_off:
            self._breathing = False
            self._rates_bpm.clear()
        elif not self._breathing and self._confidence >= s.threshold_on:
            self._breathing = True

        if self._breathing:
            self._t_last_breathing_s = t_s
            if self._t_breathing_start_s is None:
                self._t_breathing_start_s = t_s
        else:
            self._t_breathing_start_s = None

        recently_breathing = (self._t_last_breathing_s is not None
                              and t_s - self._t_last_breathing_s <= s.apnea_memory_s)
        apnea = bool(recently_breathing and np.isfinite(since_last_breath_s)
                     and since_last_breath_s >= s.apnea_s)

        rate_bpm = float(np.median(self._rates_bpm)) if self._breathing and self._rates_bpm else None
        state = STATE_BREATHING if self._breathing else STATE_NO_BREATHING
        if self._t_breathing_start_s is not None:
            breathing_for_s = t_s - self._t_breathing_start_s
        else:
            breathing_for_s = 0.0
        return BreathingState(t_s, state, self._confidence, metrics.confidence, rate_bpm,
                              self._breathing, metrics, breathing_for_s, since_last_breath_s, apnea)


# ----------------------------------------------------------------------
# Micro-Doppler chain: Fisher F-test fused with the phase ACF
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
