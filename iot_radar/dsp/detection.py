"""Breathing detection: metrics of one window, then a temporal decision.

Per window (:func:`breathing_metrics`): rate = highest peak of the spectrum
in the breathing band; the peak must be a local maximum (not the skirt of an
out-of-band component); SNR = peak / median power of a noise-only band;
concentration = share of the band energy near the peak; motion = peak-to-peak
displacement above ``motion_ptp_mm``.  Confidence = √(SNR score ×
concentration score), 0 with motion or without a genuine peak.

:class:`BreathingDetector` smooths it, applies ON / OFF thresholds, holds the
``MOTION`` state, reports the median rate and raises an apnea alert (no
inhalation for ``apnea_s`` shortly after breathing; a radar cannot tell a
breath-hold from the person leaving the beam).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from iot_radar.config import check_keys
from iot_radar.dsp.estimation import Spectrum

STATE_WARMUP = "WARMUP"
STATE_NO_BREATHING = "NO_BREATHING"
STATE_BREATHING = "BREATHING"
STATE_MOTION = "MOTION"
STATE_CODES = {STATE_WARMUP: -1, STATE_NO_BREATHING: 0, STATE_BREATHING: 1, STATE_MOTION: 2}
"""Integer code of each state (arrays, plots)."""


# ----------------------------------------------------------------------
# Per-window metrics
# ----------------------------------------------------------------------

@dataclass
class DetectionSettings:
    """Detector parameters (``vital_signs.detection``; see the comments of ``configs/radar.yaml``).

    ``snr_*`` and ``concentration_*`` map a metric to a score 0 → 1;
    ``motion_window_s`` is the recent segment also checked for motion;
    ``threshold_on`` / ``threshold_off`` are the hysteresis on the smoothed
    confidence; the apnea alert needs breathing within ``apnea_memory_s``.
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
        """From the YAML section: defaults for missing keys, unknown keys refused."""
        check_keys("vital_signs.detection", section, [f.name for f in fields(cls)])
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
    """Metrics of one window.

    ``noise_floor`` = median power of the reference band; ``displacement_ptp_mm``
    is NaN when unknown (linear demodulation); scores and ``confidence`` in [0, 1].
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
    """``value ≤ floor`` → 0, ``value ≥ good`` → 1, linear in between (NaN → 0)."""
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
    """Metrics and instantaneous confidence of one window.

    *spectrum*: of the detrended displacement; *displacement_ptp_mm*: NaN
    when not calibrated (motion then not assessed).  The noise floor excludes
    the bins near the peak and its harmonics 2 and 3: with M1 the noise band
    lies inside the pass band, and without the exclusion an empty room got
    80 dB of SNR there (detected 22–40 % of the time).
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
    near_harmonic = np.zeros_like(in_reference)
    for order in (1, 2, 3):
        near_harmonic |= np.abs(f_hz - order * f_hz[k]) <= settings.peak_halfwidth_hz
    noise_bins = in_reference & ~near_harmonic
    noise_floor = float(max(np.median(psd[noise_bins] if noise_bins.any() else psd[in_reference]), 1e-300))
    # An empty spectrum (every sample of weight 0 with M3) has no SNR at all.
    snr_db = float(10.0 * np.log10(psd[k] / noise_floor)) if psd[k] > 0 else float("-inf")

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
# Temporal decision
# ----------------------------------------------------------------------

@dataclass
class BreathingState:
    """Output of :class:`BreathingDetector` for one update.

    ``confidence`` is smoothed, ``confidence_instant`` is the window's;
    ``rate_bpm`` is the median rate (``None`` unless breathing);
    ``pause_reason`` in ``MOTION``: ``"motion"`` or ``"saturation"``.
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
    pause_reason: str | None = None


class BreathingDetector:
    """Temporal decision: smoothing, hysteresis, motion hold, apnea alert; one per stream.

    The smoothed confidence starts at 0: a detection needs ≈ −τ·ln(1 − on)
    seconds of evidence (3.7 s with the defaults), against isolated false alarms.
    """

    def __init__(self, settings: DetectionSettings, period_s: float) -> None:
        self.settings = settings
        self.period_s = float(period_s)
        self._confidence = 0.0
        self._breathing = False
        self._t_last_s: float | None = None
        self._t_motion_s: float | None = None
        self._pause_reason: str | None = None
        self._t_breathing_start_s: float | None = None
        self._t_last_breathing_s: float | None = None
        self._rates_bpm: deque[float] = deque(maxlen=max(1, int(settings.rate_median_len)))

    def warmup(self, t_s: float) -> BreathingState:
        """State while the analysis window fills."""
        return BreathingState(t_s, STATE_WARMUP, 0.0, 0.0, None, False, None)

    def update(
        self,
        metrics: BreathMetrics,
        t_s: float,
        since_last_breath_s: float = float("nan"),
        pause_reason: str | None = None,
    ) -> BreathingState:
        """Fold one window into the decision.  *since_last_breath_s* drives the
        apnea alert; *pause_reason* (e.g. ``"saturation"``) pauses like a motion."""
        s = self.settings
        if self._t_last_s is None:
            dt_s = self.period_s
        else:
            dt_s = max(0.0, t_s - self._t_last_s)
        self._t_last_s = t_s

        if metrics.motion or pause_reason is not None:
            self._t_motion_s = t_s
            self._pause_reason = "motion" if metrics.motion else pause_reason
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
                                  since_last_breath_s=since_last_breath_s, pause_reason=self._pause_reason)

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
