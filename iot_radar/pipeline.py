"""The processing chain, from IQ blocks to breathing decisions::

    Block (ADC rate) → Mixer (echo to 0 Hz) → Decimator (100 Hz) → [M3 meter]
      → M1 Doppler filter (20 Hz) → sliding window → VitalSignsProcessor
        (circle fit, arctangent, displacement, spectrum, metrics)
      → BreathingDetector → PipelineOutput (read by the dashboard)

The pipeline knows neither where the blocks come from nor how the results
are shown.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from iot_radar.dsp import estimation
from iot_radar.dsp.contamination import ContaminationMeter
from iot_radar.dsp.decimation import Decimator
from iot_radar.dsp.doppler import alias_audit, dc_group_delay_s, doppler_lowpass_decimator
from iot_radar.dsp.detection import (
    BreathingDetector,
    BreathingState,
    BreathMetrics,
    DetectionSettings,
    breathing_metrics,
)
from iot_radar.dsp.filters import bandpass, detrend, widened_band
from iot_radar.dsp.mixer import Mixer
from iot_radar.dsp.phase import (
    DemodulationResult,
    demodulate,
    phase_to_displacement_m,
    remove_common_rotation,
)
from iot_radar.dsp.spectral import compute_single_column, frequency_axis, get_window
from iot_radar.measurement import (
    STATE_DONE,
    STATE_IDLE,
    STATE_RUNNING,
    MeasurementResult,
    MeasurementSettings,
    MeasurementStatus,
    measurement_decision,
)
from iot_radar.config import check_keys, check_radar_config
from iot_radar.physics import SPEED_OF_LIGHT

if TYPE_CHECKING:  # only for the type hints: the pipeline does not depend on the acquisition
    from iot_radar.acquisition.sources import Block, Source

logger = logging.getLogger(__name__)


_ZERO_PAD = 8
"""Zero-padding factor of the displacement spectrum."""


@dataclass
class VitalSignsSettings:
    """``vital_signs`` section (see the comments of ``configs/radar.yaml``).

    ``lo_drift_compensation`` is off: the Pluto TX and RX share one LO, and
    the ramp fitted on a window would also take a slow chest movement.
    """

    window_s: float = 20.0
    update_period_s: float = 0.5
    breathing_band_hz: tuple[float, float] = (0.1, 0.5)
    reference_band_hz: tuple[float, float] = (2.5, 5.0)
    min_arc_rad: float = 0.3
    max_circle_residual: float = 0.3
    lo_drift_compensation: bool = False
    bandpass_order: int = 4
    detection: DetectionSettings = field(default_factory=DetectionSettings)

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "VitalSignsSettings":
        """From the YAML section (defaults for missing keys, unknown keys refused)."""
        check_keys("vital_signs", section, [f.name for f in fields(cls)])
        defaults = cls()
        return cls(
            window_s=float(section.get("window_s", defaults.window_s)),
            update_period_s=float(section.get("update_period_s", defaults.update_period_s)),
            breathing_band_hz=tuple(section.get("breathing_band_hz", defaults.breathing_band_hz)),
            reference_band_hz=tuple(section.get("reference_band_hz", defaults.reference_band_hz)),
            min_arc_rad=float(section.get("min_arc_rad", defaults.min_arc_rad)),
            max_circle_residual=float(section.get("max_circle_residual", defaults.max_circle_residual)),
            lo_drift_compensation=bool(section.get("lo_drift_compensation", defaults.lo_drift_compensation)),
            bandpass_order=int(section.get("bandpass_order", defaults.bandpass_order)),
            detection=DetectionSettings.from_config(section.get("detection", {})),
        )


@dataclass
class MotionRobustnessSettings:
    """``motion_robustness`` section: M1 (Doppler filter) and M3 (contamination weights).

    Both off by default (former behaviour).  See the comments of
    ``configs/radar.yaml`` for each parameter.
    """

    doppler_rate_hz: float = 100.0
    doppler_filter_enabled: bool = False
    cutoff_hz: float = 1.5
    transition_hz: float = 1.0
    stopband_attenuation_db: float = 60.0
    noise_band_hz: tuple[float, float] = (0.1, 1.4)
    alias_band_hz: float = 1.0
    min_alias_attenuation_db: float = 60.0
    contamination_enabled: bool = False
    contamination_block_s: float = 1.0
    out_of_band_start_hz: float = 2.0
    in_band_hz: tuple[float, float] = (0.1, 1.0)
    weight_midpoint_db: float = 9.0
    weight_slope_db: float = 2.0
    contaminated_below: float = 0.5
    activity_band_hz: tuple[float, float] = (3.0, 28.0)
    activity_reference_hz: tuple[float, float] = (30.0, 38.0)
    activity_threshold_db: float = 6.0
    activity_lead_s: float = 10.0
    excursion_factor: float = 2.5
    excursion_floor_mm: float = 12.0
    min_segment_s: float = 5.0
    pause_on_saturation: bool = True

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "MotionRobustnessSettings":
        """From the YAML section (defaults for missing keys, unknown keys refused)."""
        d = cls()
        check_keys("motion_robustness", section,
                   ["doppler_rate_hz", "doppler_filter", "alias_check", "contamination_weighting"])
        doppler = section.get("doppler_filter", {})
        alias = section.get("alias_check", {})
        weighting = section.get("contamination_weighting", {})
        check_keys("motion_robustness.doppler_filter", doppler,
                   ["enabled", "cutoff_hz", "transition_hz", "stopband_attenuation_db", "noise_band_hz"])
        check_keys("motion_robustness.alias_check", alias, ["band_hz", "min_attenuation_db"])
        check_keys("motion_robustness.contamination_weighting", weighting,
                   ["enabled", "block_s", "out_of_band_start_hz", "in_band_hz", "weight_midpoint_db",
                    "weight_slope_db", "contaminated_below", "activity_band_hz", "activity_reference_hz",
                    "activity_threshold_db", "activity_lead_s", "excursion_factor", "excursion_floor_mm",
                    "min_segment_s", "pause_on_saturation"])
        return cls(
            doppler_rate_hz=float(section.get("doppler_rate_hz", d.doppler_rate_hz)),
            doppler_filter_enabled=bool(doppler.get("enabled", d.doppler_filter_enabled)),
            cutoff_hz=float(doppler.get("cutoff_hz", d.cutoff_hz)),
            transition_hz=float(doppler.get("transition_hz", d.transition_hz)),
            stopband_attenuation_db=float(doppler.get("stopband_attenuation_db", d.stopband_attenuation_db)),
            noise_band_hz=tuple(doppler.get("noise_band_hz", d.noise_band_hz)),
            alias_band_hz=float(alias.get("band_hz", d.alias_band_hz)),
            min_alias_attenuation_db=float(alias.get("min_attenuation_db", d.min_alias_attenuation_db)),
            contamination_enabled=bool(weighting.get("enabled", d.contamination_enabled)),
            contamination_block_s=float(weighting.get("block_s", d.contamination_block_s)),
            out_of_band_start_hz=float(weighting.get("out_of_band_start_hz", d.out_of_band_start_hz)),
            in_band_hz=tuple(weighting.get("in_band_hz", d.in_band_hz)),
            weight_midpoint_db=float(weighting.get("weight_midpoint_db", d.weight_midpoint_db)),
            weight_slope_db=float(weighting.get("weight_slope_db", d.weight_slope_db)),
            contaminated_below=float(weighting.get("contaminated_below", d.contaminated_below)),
            activity_band_hz=tuple(weighting.get("activity_band_hz", d.activity_band_hz)),
            activity_reference_hz=tuple(weighting.get("activity_reference_hz", d.activity_reference_hz)),
            activity_threshold_db=float(weighting.get("activity_threshold_db", d.activity_threshold_db)),
            activity_lead_s=float(weighting.get("activity_lead_s", d.activity_lead_s)),
            excursion_factor=float(weighting.get("excursion_factor", d.excursion_factor)),
            excursion_floor_mm=float(weighting.get("excursion_floor_mm", d.excursion_floor_mm)),
            min_segment_s=float(weighting.get("min_segment_s", d.min_segment_s)),
            pause_on_saturation=bool(weighting.get("pause_on_saturation", d.pause_on_saturation)),
        )


@dataclass
class WindowAnalysis:
    """Everything computed on one analysis window.

    Displacements in mm (arbitrary units if not ``demodulation.calibrated``),
    positive away from the radar; ``iq`` is the window demodulated (for the
    constellation).  M3 only: ``weights`` per sample, ``mean_weight`` (it
    multiplies the confidence), ``contaminated_fraction`` (weight below
    ``contaminated_below``), ``target_motion_fraction`` (anomalies of the target).
    """

    t_s: np.ndarray
    displacement_mm: np.ndarray
    breath_mm: np.ndarray
    spectrum_f_hz: np.ndarray
    spectrum_psd: np.ndarray
    metrics: BreathMetrics
    cycles: estimation.BreathCycles
    iq: np.ndarray
    lo_drift_hz: float
    demodulation: DemodulationResult
    weights: np.ndarray | None = None
    mean_weight: float = 1.0
    contaminated_fraction: float = 0.0
    target_motion_fraction: float = 0.0


@dataclass
class _InBandAnomalies:
    """M3 cuts of one window (:meth:`VitalSignsProcessor._cut_in_band_anomalies`)."""

    weights: np.ndarray          # weights, 0 on the cut samples
    own: np.ndarray              # samples in anomalies attributed to the target
    recent_ptp_mm: float         # target's motion over the last motion_window_s (mm)


class VitalSignsProcessor:
    """Analysis of one window of slow-time IQ.

    Stateless: each call re-analyses the latest window, which follows slow
    scene changes and allows zero-phase filters.
    """

    def __init__(self, f_s_hz: float, wavelength_m: float, settings: VitalSignsSettings,
                 motion: MotionRobustnessSettings | None = None) -> None:
        self.f_s_hz = float(f_s_hz)
        self.wavelength_m = float(wavelength_m)
        self.settings = settings
        self.motion = motion or MotionRobustnessSettings()
        self.window_length = int(round(settings.window_s * self.f_s_hz))
        nyquist_hz = self.f_s_hz / 2.0
        for name, band in (("breathing_band_hz", settings.breathing_band_hz),
                           ("reference_band_hz", settings.reference_band_hz)):
            if not 0 < band[0] < band[1] < nyquist_hz:
                raise ValueError(f"{name}={band} incompatible with a {self.f_s_hz} Hz slow time.")
        if settings.window_s * settings.breathing_band_hz[0] < 2.0:
            logger.warning(
                "The %.1f s window holds fewer than 2 periods of the slowest breath "
                "(%.2f Hz): estimates near the band edge will be unreliable.",
                settings.window_s, settings.breathing_band_hz[0],
            )
        self._n_fft = int(2 ** np.ceil(np.log2(self.window_length * _ZERO_PAD)))
        # Frequencies of the weighted Lomb-Scargle periodogram (M3): same
        # interpolation step as the zero-padded periodogram, up to the noise bands.
        step_hz = 1.0 / (settings.window_s * _ZERO_PAD)
        top_hz = min(nyquist_hz, max(settings.reference_band_hz[1], self.motion.noise_band_hz[1]) + 0.5)
        self._ls_frequencies_hz = np.arange(step_hz, top_hz, step_hz)
        logger.info(
            "VitalSignsProcessor — %.2f Hz slow time, %d-sample window (%.1f s), "
            "λ=%.4f m",
            self.f_s_hz, self.window_length, settings.window_s, self.wavelength_m,
        )

    def _recent_peak_to_peak_mm(self, z: np.ndarray) -> float:
        """Peak-to-peak displacement (mm) of the last ``motion_window_s``, with its own
        circle fit (NaN if it cannot be demodulated)."""
        n = int(round(self.settings.detection.motion_window_s * self.f_s_hz))
        if n < 4 or n >= z.size:
            return float("nan")
        result = demodulate(z[-n:], self.settings.min_arc_rad, self.settings.max_circle_residual)
        if not result.calibrated:
            return float("nan")
        return float(np.ptp(phase_to_displacement_m(result.phase_rad, self.wavelength_m)) * 1e3)

    def analyse(self, z: np.ndarray, weights: np.ndarray | None = None,
                disturber_active: np.ndarray | None = None) -> WindowAnalysis:
        """Analyse one window.

        With M3, *weights* (one per sample) give the weighted Lomb-Scargle
        spectrum and scale the confidence; the in-band anomalies are cut and
        only the target's pause the measurement (*disturber_active*: someone
        moving out of band, per sample).
        """
        s = self.settings
        weighted = self.motion.contamination_enabled and weights is not None
        z = np.asarray(z, dtype=np.complex128)
        t_s = np.arange(z.size) / self.f_s_hz

        lo_drift_hz = 0.0
        if s.lo_drift_compensation:
            z, lo_drift_hz = remove_common_rotation(z, self.f_s_hz)
        result, displacement_mm = self._demodulate(z)
        if weighted:
            anomalies = self._cut_in_band_anomalies(displacement_mm, result.calibrated, weights, disturber_active)
        # Detrend for the spectrum and the motion check only: it biases the breathing amplitude.
        detrended_mm = detrend(displacement_mm)

        breath_mm = bandpass(displacement_mm, self.f_s_hz,
                             widened_band(s.breathing_band_hz, self.f_s_hz), s.bandpass_order)

        if weighted:
            w, displacement_ptp_mm = anomalies.weights, anomalies.recent_ptp_mm
            spectrum = estimation.weighted_lomb_scargle(t_s, estimation.remove_segment_offsets(detrended_mm, w),
                                                        w, self._ls_frequencies_hz)
        else:
            spectrum = estimation.periodogram(detrended_mm, self.f_s_hz, n_fft=self._n_fft)
            # Largest peak-to-peak of the window and of its recent part (NaN if not in mm).
            whole_window_ptp_mm = float(np.ptp(detrended_mm)) if result.calibrated else float("nan")
            measurable = [v for v in (whole_window_ptp_mm, self._recent_peak_to_peak_mm(z)) if np.isfinite(v)]
            displacement_ptp_mm = max(measurable) if measurable else float("nan")
        reference_band_hz = self.motion.noise_band_hz if self.motion.doppler_filter_enabled else s.reference_band_hz
        metrics = breathing_metrics(spectrum, displacement_ptp_mm, s.breathing_band_hz,
                                    reference_band_hz, s.detection)
        mean_weight, contaminated_fraction, target_motion_fraction = 1.0, 0.0, 0.0
        if weighted:
            mean_weight = float(w.mean())
            contaminated_fraction = float(np.mean(w < self.motion.contaminated_below))
            target_motion_fraction = float(anomalies.own.mean())
            metrics = replace(metrics, confidence=metrics.confidence * mean_weight)

        cycles = estimation.breath_cycles(breath_mm, self.f_s_hz, s.breathing_band_hz)

        logger.debug(
            "Window — rate=%s Hz, SNR=%.1f dB, concentration=%.2f, motion=%s, confidence=%.2f, demod=%s",
            f"{metrics.rate_hz:.3f}" if metrics.rate_hz else "n/a", metrics.snr_db,
            metrics.concentration, metrics.motion, metrics.confidence, result.method,
        )
        return WindowAnalysis(
            t_s=t_s,
            displacement_mm=displacement_mm,
            breath_mm=breath_mm,
            spectrum_f_hz=spectrum.f_hz,
            spectrum_psd=spectrum.psd,
            metrics=metrics,
            cycles=cycles,
            iq=z,
            lo_drift_hz=lo_drift_hz,
            demodulation=result,
            weights=w if weighted else None,
            mean_weight=mean_weight,
            contaminated_fraction=contaminated_fraction,
            target_motion_fraction=target_motion_fraction,
        )

    def _demodulate(self, z: np.ndarray) -> tuple[DemodulationResult, np.ndarray]:
        """Demodulation and displacement in mm, mean removed."""
        s = self.settings
        result = demodulate(z, s.min_arc_rad, s.max_circle_residual)
        if result.calibrated:
            displacement_mm = phase_to_displacement_m(result.phase_rad, self.wavelength_m) * 1e3
        else:
            displacement_mm = result.phase_rad  # arbitrary units (linear fallback)
        return result, displacement_mm - displacement_mm.mean()

    def _cut_in_band_anomalies(self, displacement_mm: np.ndarray, calibrated: bool, weights: np.ndarray,
                               disturber_active: np.ndarray | None) -> "_InBandAnomalies":
        """M3: cut the in-band anomalies out of the weights; measure the target's own motion.

        Anomaly: local peak-to-peak (3 s, not detrended) above
        ``motion_ptp_mm`` (30 mm; breathing moves the chest by 5–15 mm).
        Also cut: excursions above ``excursion_factor`` × the median local
        peak-to-peak, and away from any activity above ``excursion_floor_mm``
        too (keeps the breathing next to an apnea); segments left shorter than
        ``min_segment_s`` (they gave slow false peaks with walkers).

        An anomaly preceded by out-of-band activity within ``activity_lead_s``
        is a passer-by's (a person reaches the breathing band by slowing down
        through higher Doppler); the others are the target's and pause the
        measurement.  Limits: a target moving while a passer-by is active is
        not paused; someone starting to walk next to the target is taken for it.

        Returns the weights with the cuts, the target mask and the largest
        local peak-to-peak (mm) of the recent part outside the passers-by
        (NaN when the window is not calibrated).
        """
        w = np.asarray(weights, dtype=np.float64).copy()
        if not calibrated:
            return _InBandAnomalies(w, np.zeros(w.size, dtype=bool), float("nan"))
        s = self.settings.detection
        span = max(3, int(round(3.0 * self.f_s_hz)))
        local_ptp_mm = (maximum_filter1d(displacement_mm, span, mode="nearest")
                        - minimum_filter1d(displacement_mm, span, mode="nearest"))
        # No floor next to a passer-by: any excursion above factor × median is suspect.
        relative_mm = self.motion.excursion_factor * float(np.median(local_ptp_mm))
        excursion_mm = np.full(local_ptp_mm.size, max(relative_mm, self.motion.excursion_floor_mm))
        if disturber_active is not None:
            near_activity = maximum_filter1d(np.asarray(disturber_active, dtype=np.uint8), span, mode="nearest") > 0
            excursion_mm[near_activity] = relative_mm
        cut = local_ptp_mm > np.minimum(s.motion_ptp_mm, excursion_mm)
        anomaly = local_ptp_mm > s.motion_ptp_mm
        attributed = np.zeros_like(anomaly)
        if disturber_active is not None:
            lead = int(round(self.motion.activity_lead_s * self.f_s_hz))
            edges = np.flatnonzero(np.diff(np.r_[0, anomaly.astype(np.int8), 0]))
            for start, stop in zip(edges[::2], edges[1::2]):
                attributed[start:stop] = bool(np.any(disturber_active[max(0, start - lead):start + 1]))
        w[cut] = 0.0
        # A short segment cannot hold a breath; its own offset would make a slow false peak.
        min_length = int(round(self.motion.min_segment_s * self.f_s_hz))
        edges = np.flatnonzero(np.diff(np.r_[0, (w > 0).astype(np.int8), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start < min_length and (start > 0 or stop < w.size) and cut.any():
                w[start:stop] = 0.0
        n_recent = int(round(s.motion_window_s * self.f_s_hz))
        ptp_mm = float(np.max(local_ptp_mm[-n_recent:], where=~attributed[-n_recent:], initial=0.0))
        return _InBandAnomalies(w, anomaly & ~attributed, ptp_mm)


@dataclass
class ContaminationStatus:
    """M3 indicators of one output: κ and weight of the latest block, mean weight and
    contaminated share of the window, ADC saturation within ``motion_hold_s``."""

    index_db: float
    weight: float
    mean_weight: float
    contaminated_fraction: float
    saturated: bool


@dataclass
class PipelineOutput:
    """One output of :class:`VitalSignsPipeline` (every ``update_period_s``).

    ``discontinuity``: samples were lost since the previous output (processing
    restarted); ``analysis`` is ``None`` and the state ``WARMUP`` while the
    window fills; ``contamination`` is ``None`` without M3.
    """

    update_index: int
    t_s: float
    discontinuity: bool
    analysis: WindowAnalysis | None
    breathing: BreathingState
    micro_doppler_column_db: np.ndarray | None
    contamination: ContaminationStatus | None = None
    measurement: MeasurementStatus | None = None


class VitalSignsPipeline:
    """Pipeline from IQ blocks to breathing decisions.

    *input_rate_hz* / *slow_time_rate_hz* must be an integer of primes ≤ 13;
    *tx_offset_hz* is the frequency of the echo in the blocks (0 in pure CW);
    *warmup_s* is dropped after a restart; *adc_full_scale* (2048 on the
    Pluto, ``None`` for converted recordings) detects the saturation with M3.

    A block with ``overflow`` set, or whose ``sample_start`` does not follow
    the previous block, means lost samples (phase unknown): everything is
    reset, the next output has ``discontinuity=True`` and the window refills.
    """

    def __init__(
        self,
        settings: VitalSignsSettings,
        input_rate_hz: float,
        tx_offset_hz: float,
        wavelength_m: float,
        slow_time_rate_hz: float = 20.0,
        warmup_s: float = 1.0,
        micro_doppler_window_s: float = 3.2,
        micro_doppler_n_fft: int = 256,
        micro_doppler_window: str = "hann",
        motion: MotionRobustnessSettings | None = None,
        adc_full_scale: float | None = None,
        measurement: MeasurementSettings | None = None,
    ) -> None:
        decimation_ratio = input_rate_hz / slow_time_rate_hz
        if abs(decimation_ratio - round(decimation_ratio)) > 1e-9:
            raise ValueError(
                f"The input rate {input_rate_hz} Hz is not a whole multiple of the "
                f"slow-time rate {slow_time_rate_hz} Hz."
            )
        self.settings = settings
        self.input_rate_hz = float(input_rate_hz)
        self.slow_time_rate_hz = float(slow_time_rate_hz)
        self.motion = motion or MotionRobustnessSettings()
        self._mixer = Mixer(input_rate_hz, tx_offset_hz)
        max_frequency_hz = settings.reference_band_hz[1]
        self._build_decimation(int(round(decimation_ratio)), max_frequency_hz)
        if self.motion.doppler_filter_enabled:  # the FIR must fill before its output is valid
            warmup_s = max(warmup_s, 2.0 * self.doppler_delay_s)
        self._processor = VitalSignsProcessor(slow_time_rate_hz, wavelength_m, settings, self.motion)
        self.adc_full_scale = adc_full_scale
        self._contamination: ContaminationMeter | None = None
        if self.motion.contamination_enabled:
            m = self.motion
            self._contamination = ContaminationMeter(m.doppler_rate_hz, m.contamination_block_s,
                                                     m.out_of_band_start_hz, m.in_band_hz,
                                                     m.weight_midpoint_db, m.weight_slope_db,
                                                     activity_band_hz=m.activity_band_hz,
                                                     reference_band_hz=m.activity_reference_hz)
            self._doppler_per_slow = int(round(m.doppler_rate_hz / slow_time_rate_hz))
            self._delay_doppler_samples = int(round(self.doppler_delay_s * m.doppler_rate_hz))
        self._warmup_samples = int(round(warmup_s * slow_time_rate_hz))
        self._update_samples = max(1, int(round(settings.update_period_s * slow_time_rate_hz)))
        self._micro_doppler_samples = int(round(micro_doppler_window_s * slow_time_rate_hz))
        self._micro_doppler_window = get_window(micro_doppler_window, self._micro_doppler_samples)
        self._micro_doppler_n_fft = int(micro_doppler_n_fft)
        self.micro_doppler_f_hz = frequency_axis(self._micro_doppler_n_fft, self.slow_time_rate_hz)

        history_length = max(self._processor.window_length, self._micro_doppler_samples)
        self._history: deque[complex] = deque(maxlen=history_length)
        self._weights: deque[float] = deque(maxlen=history_length)   # M3: weight of each slow-time sample
        self._active: deque[bool] = deque(maxlen=history_length)     # M3: someone moving out of band
        self._t_saturation_s: float | None = None
        self._saturated_since_output = False
        self.measurement_settings = measurement or MeasurementSettings()
        self._measurement: dict[str, Any] | None = None       # running measurement (M4)
        self._measurement_result: MeasurementResult | None = None
        self._measurement_completed = False
        self._next_sample_start: int | None = None
        self._slow_samples_total = 0
        self._update_index = 0
        self._discontinuity_pending = False
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], input_rate_hz: float, tx_offset_hz: float,
                    adc_full_scale: float | None = None) -> "VitalSignsPipeline":
        """From the configuration; ``ValueError`` on any unknown key."""
        check_radar_config(cfg)
        slow_time = cfg.get("slow_time", {})
        view = cfg.get("micro_doppler_view", {})
        return cls(
            settings=VitalSignsSettings.from_config(cfg.get("vital_signs", {})),
            input_rate_hz=input_rate_hz,
            tx_offset_hz=tx_offset_hz,
            wavelength_m=SPEED_OF_LIGHT / float(cfg["sdr"]["center_frequency_hz"]),
            slow_time_rate_hz=float(slow_time.get("rate_hz", 20.0)),
            warmup_s=float(slow_time.get("warmup_s", 1.0)),
            micro_doppler_window_s=float(view.get("window_s", 3.2)),
            micro_doppler_n_fft=int(view.get("n_fft", 256)),
            micro_doppler_window=str(view.get("window", "hann")),
            motion=MotionRobustnessSettings.from_config(cfg.get("motion_robustness", {})),
            adc_full_scale=adc_full_scale,
            measurement=MeasurementSettings.from_config(cfg.get("measurement", {})),
        )

    def reset(self) -> None:
        """Restart the oscillator, filters, history and detector."""
        self._mixer.reset()
        self._decimator.reset()
        if self._doppler_stage is not None:
            self._doppler_stage.reset()
        self._history.clear()
        self._weights.clear()
        self._active.clear()
        self._slow_since_reset = 0
        if self._contamination is not None:
            self._contamination.reset()
        self._samples_in_history = 0
        self._samples_since_update = 0
        self._warmup_remaining = self._warmup_samples
        self._detector = BreathingDetector(self.settings.detection, self.settings.update_period_s)

    def process_block(self, block: Block) -> list[PipelineOutput]:
        """Process one block; return the new outputs."""
        samples = block.samples[0]
        if self._next_sample_start is not None and (
            block.overflow or block.sample_start != self._next_sample_start
        ):
            logger.warning("Stream discontinuity before sample %d — processing restarted", block.sample_start)
            self.reset()
            self._discontinuity_pending = True
            if self._measurement is not None:
                self._finish_measurement(discontinuity=True)
        self._next_sample_start = block.sample_start + samples.size

        saturated = False
        if self._contamination is not None and self.adc_full_scale is not None:
            peak = max(float(np.max(np.abs(samples.real))), float(np.max(np.abs(samples.imag))))
            saturated = peak >= self.adc_full_scale - 1
            if saturated:
                self._t_saturation_s = self._slow_samples_total / self.slow_time_rate_hz
                self._saturated_since_output = True

        if self._doppler_stage is None:
            slow_samples = self._decimator(self._mixer(samples))
        else:
            doppler_iq = self._decimator(self._mixer(samples))
            if self._contamination is not None:
                # Before M1, which removes the disturbers.
                self._contamination.process(doppler_iq, saturated)
            slow_samples = self._doppler_stage(doppler_iq)
        outputs: list[PipelineOutput] = []
        for value in slow_samples:
            self._slow_samples_total += 1
            self._slow_since_reset += 1
            if self._warmup_remaining > 0:  # decimation filters still settling
                self._warmup_remaining -= 1
                continue
            self._history.append(complex(value))
            weight, active, saturated = self._sample_contamination()
            self._weights.append(weight)
            self._active.append(active)
            if self._measurement is not None:
                self._collect_measurement(complex(value), weight, active, saturated)
            self._samples_in_history += 1
            self._samples_since_update += 1
            if self._samples_since_update == self._update_samples:
                self._samples_since_update = 0
                outputs.append(self._make_output())
        return outputs

    def _build_decimation(self, decimation_factor: int, max_frequency_hz: float) -> None:
        """Decimation: one Chebyshev cascade without M1 / M3; otherwise cut at
        ``doppler_rate_hz`` (complex128 in between: the cut alone changes
        nothing), the last stage being M1 or the Chebyshev stage.  The
        aliasing is checked against ``min_alias_attenuation_db``."""
        m = self.motion
        self._doppler_stage: Decimator | None = None
        self.doppler_delay_s = 0.0
        if not (m.doppler_filter_enabled or m.contamination_enabled):
            self._decimator = Decimator(self.input_rate_hz, decimation_factor, max_frequency_hz)
            stages = self._decimator.stages
        else:
            to_doppler = self.input_rate_hz / m.doppler_rate_hz
            to_slow = m.doppler_rate_hz / self.slow_time_rate_hz
            if abs(to_doppler - round(to_doppler)) > 1e-9 or abs(to_slow - round(to_slow)) > 1e-9:
                raise ValueError(f"The Doppler rate {m.doppler_rate_hz} Hz must divide the input rate and be a "
                                 f"multiple of the slow-time rate {self.slow_time_rate_hz} Hz.")
            self._decimator = Decimator(self.input_rate_hz, int(round(to_doppler)), max_frequency_hz,
                                        output_dtype=np.complex128)
            if m.doppler_filter_enabled:
                self._doppler_stage, self.doppler_delay_s = doppler_lowpass_decimator(
                    m.doppler_rate_hz, int(round(to_slow)), m.cutoff_hz, m.transition_hz, m.stopband_attenuation_db)
            else:
                self._doppler_stage = Decimator(m.doppler_rate_hz, int(round(to_slow)), max_frequency_hz)
                self.doppler_delay_s = dc_group_delay_s(self._doppler_stage.stages)
            stages = self._decimator.stages + self._doppler_stage.stages
        worst = min(stage.worst_attenuation_db for stage in alias_audit(stages, m.alias_band_hz))
        self.alias_attenuation_db = worst
        if worst < m.min_alias_attenuation_db:
            logger.warning("Aliasing into ±%.1f Hz attenuated by only %.1f dB (required %.1f dB)",
                           m.alias_band_hz, worst, m.min_alias_attenuation_db)
        else:
            logger.info("Aliasing into ±%.1f Hz attenuated by at least %.1f dB", m.alias_band_hz, worst)

    def run(self, source: Source) -> Iterator[PipelineOutput]:
        """Process every block of *source*, then close it."""
        try:
            while True:
                block = source.read_block()
                if block is None:
                    break
                yield from self.process_block(block)
        finally:
            source.close()

    def _sample_contamination(self) -> tuple[float, bool, bool]:
        """M3 ``(weight, someone moving out of band, saturated)`` of the newest slow-time sample.

        Taken from the contamination block of the samples it comes from,
        ``doppler_delay_s`` earlier; ``(1, False, False)`` without M3.
        """
        if self._contamination is None:
            return 1.0, False, False
        source_index = self._slow_since_reset * self._doppler_per_slow - self._delay_doppler_samples
        block = self._contamination.block_at(max(source_index, 0))
        if block is None:
            return 1.0, False, False
        return block.weight, block.activity_db >= self.motion.activity_threshold_db, block.saturated

    # -- M4: measurement mode ------------------------------------------------

    def start_measurement(self, duration_s: float | None = None) -> None:
        """Start (or restart) a measurement of *duration_s* (``measurement.duration_s``).

        Call it when the silence starts.  Only the IQ received from now on
        counts: the samples already in the M1 filter are skipped.
        """
        duration_s = float(duration_s or self.measurement_settings.duration_s)
        self._measurement = {
            "duration_s": duration_s,
            "n_samples": int(round(duration_s * self.slow_time_rate_hz)),
            "skip": int(round(self.doppler_delay_s * self.slow_time_rate_hz)),
            "t_start_s": None,
            "z": [], "weights": [], "active": [], "saturated": [],
        }
        logger.info("Measurement started (%.0f s)", duration_s)

    @property
    def measuring(self) -> bool:
        """``True`` while a measurement runs."""
        return self._measurement is not None

    def _collect_measurement(self, value: complex, weight: float, active: bool, saturated: bool) -> None:
        m = self._measurement
        if m["skip"] > 0:
            m["skip"] -= 1
            return
        if m["t_start_s"] is None:
            m["t_start_s"] = self._slow_samples_total / self.slow_time_rate_hz
        m["z"].append(value)
        m["weights"].append(weight)
        m["active"].append(active)
        m["saturated"].append(saturated)
        if len(m["z"]) >= m["n_samples"]:
            self._finish_measurement()

    def _finish_measurement(self, discontinuity: bool = False) -> None:
        """Analyse the measurement once and decide (cut short on a discontinuity)."""
        m, self._measurement = self._measurement, None
        t_end_s = self._slow_samples_total / self.slow_time_rate_hz
        t_start_s = m["t_start_s"] if m["t_start_s"] is not None else t_end_s
        n = len(m["z"])
        saturated_fraction = float(np.mean(m["saturated"])) if n else 0.0
        activity_fraction = float(np.mean(m["active"])) if n else 0.0
        analysis, confidence, rate_bpm, target_motion = None, 0.0, None, False
        mean_weight, contaminated_fraction = 1.0, 0.0
        if not discontinuity:
            settings = replace(self.settings, window_s=m["duration_s"])
            processor = VitalSignsProcessor(self.slow_time_rate_hz, self._processor.wavelength_m, settings,
                                            self.motion)
            weights = np.array(m["weights"]) if self._contamination is not None else None
            analysis = processor.analyse(np.array(m["z"]), weights, np.array(m["active"], dtype=bool))
            confidence = analysis.metrics.confidence
            mean_weight, contaminated_fraction = analysis.mean_weight, analysis.contaminated_fraction
            target_motion = analysis.metrics.motion or analysis.target_motion_fraction > 0.0
            if analysis.metrics.rate_hz is not None:
                rate_bpm = 60.0 * analysis.metrics.rate_hz
        detection = self.settings.detection
        decision, reasons = measurement_decision(
            confidence, 1.0 - contaminated_fraction, saturated_fraction > 0.0, target_motion, discontinuity,
            self.measurement_settings.min_valid_fraction, detection.threshold_on, detection.threshold_off)
        self._measurement_result = MeasurementResult(
            t_start_s, t_end_s, decision, reasons, rate_bpm if decision == "breathing" else None, confidence,
            mean_weight, contaminated_fraction, saturated_fraction, activity_fraction, target_motion,
            discontinuity, analysis)
        self._measurement_completed = True
        logger.info("Measurement done: %s%s, confidence %.2f, contaminated %.0f %%", decision,
                    f" {rate_bpm:.1f} /min" if decision == "breathing" else f" {reasons}", confidence,
                    100 * contaminated_fraction)

    def _measurement_status(self) -> MeasurementStatus:
        """``done`` once after a completion, then ``running`` or ``idle``."""
        last = self._measurement_result
        if self._measurement_completed:
            self._measurement_completed = False
            duration_s = last.t_end_s - last.t_start_s
            return MeasurementStatus(STATE_DONE, duration_s, duration_s, last)
        if self._measurement is not None:
            m = self._measurement
            return MeasurementStatus(STATE_RUNNING, len(m["z"]) / self.slow_time_rate_hz, m["duration_s"], last)
        duration_s = last.t_end_s - last.t_start_s if last else self.measurement_settings.duration_s
        return MeasurementStatus(STATE_IDLE, duration_s if last else 0.0, duration_s, last)

    def _make_output(self) -> PipelineOutput:
        """Analyse the latest window and build one output."""
        t_s = self._slow_samples_total / self.slow_time_rate_hz
        history = np.array(self._history, dtype=np.complex128)

        micro_doppler_column_db = None
        if self._samples_in_history >= self._micro_doppler_samples:
            segment = history[-self._micro_doppler_samples:]
            # Mean removed: fine for a magnitude display, never before the phase.
            column = compute_single_column(segment - segment.mean(), self.slow_time_rate_hz,
                                           self._micro_doppler_window, self._micro_doppler_n_fft)
            micro_doppler_column_db = column.power_db

        analysis = None
        breathing = self._detector.warmup(t_s)
        window = self._processor.window_length
        saturated = (self._t_saturation_s is not None
                     and t_s - self._t_saturation_s <= self.settings.detection.motion_hold_s)
        if self._samples_in_history >= window:
            if self._contamination is None:
                analysis = self._processor.analyse(history[-window:])
                breathing = self._detector.update(analysis.metrics, t_s, analysis.cycles.since_last_s)
            else:
                weights = np.array(self._weights, dtype=np.float64)[-window:]
                active = np.array(self._active, dtype=bool)[-window:]
                analysis = self._processor.analyse(history[-window:], weights, active)
                # Held by the detector for motion_hold_s.
                pause_reason = ("saturation" if self._saturated_since_output and self.motion.pause_on_saturation
                                else None)
                breathing = self._detector.update(analysis.metrics, t_s, analysis.cycles.since_last_s,
                                                  pause_reason)

        contamination = None
        if self._contamination is not None:
            latest = self._contamination.latest
            contamination = ContaminationStatus(
                index_db=latest.index_db if latest else float("nan"),
                weight=latest.weight if latest else 1.0,
                mean_weight=analysis.mean_weight if analysis else float("nan"),
                contaminated_fraction=analysis.contaminated_fraction if analysis else float("nan"),
                saturated=saturated,
            )

        self._saturated_since_output = False
        self._update_index += 1
        output = PipelineOutput(self._update_index, t_s, self._discontinuity_pending,
                                analysis, breathing, micro_doppler_column_db, contamination,
                                self._measurement_status())
        self._discontinuity_pending = False
        return output
