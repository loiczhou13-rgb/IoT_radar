"""Assembly of the phase-demodulation chain: IQ blocks in, results out.

Place in the chain: everything between the acquisition and the display::

    Block (ADC rate) → Mixer (echo to 0 Hz) → Decimator (slow time, ~20 Hz)
      → sliding window → VitalSignsProcessor: LO-drift derotation, circle fit
        (DC compensation), arctangent / DACM, displacement, band-pass,
        spectrum, breathing metrics
      → BreathingDetector (smoothing, hysteresis, motion, apnea)
      → PipelineOutput (dataclasses, consumed by the dashboard)

The pipeline knows neither where the blocks come from (any object with the
fields of :class:`iot_radar.acquisition.sources.Block`) nor how the results
are displayed.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from iot_radar.dsp import estimation
from iot_radar.dsp.decimation import Decimator
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
from iot_radar.physics import SPEED_OF_LIGHT

if TYPE_CHECKING:  # only for the type hints: the pipeline does not depend on the acquisition
    from iot_radar.acquisition.sources import Block, Source

logger = logging.getLogger(__name__)


_ZERO_PAD = 8
"""Zero-padding factor of the displacement spectrum (smoother peak location)."""


@dataclass
class VitalSignsSettings:
    """Parameters of the window analysis (``vital_signs`` section of the configuration).

    Attributes
    ----------
    window_s : float
        Length of the analysis window (s); ≥ 2 periods of the slowest breath.
    update_period_s : float
        Time between two analyses (s).
    breathing_band_hz, heart_band_hz, reference_band_hz : tuple[float, float]
        Breathing, heartbeat and noise-only bands (Hz).
    demodulation : str
        ``"arctan"``, ``"dacm"`` or ``"linear"``.
    dc_compensation : str
        ``"circle"`` (recommended), ``"mean"`` or ``"none"``.
    min_arc_rad, max_circle_residual : float
        Validity limits of the circle fit.
    lo_drift_compensation : bool
        Remove a common phase ramp (residual TX/RX LO offset) first.
    bandpass_order : int
        Butterworth order of the band-pass filters.
    detection : DetectionSettings
        Parameters of the breathing detector.
    """

    window_s: float = 20.0
    update_period_s: float = 0.5
    breathing_band_hz: tuple[float, float] = (0.1, 0.5)
    heart_band_hz: tuple[float, float] = (0.8, 2.0)
    reference_band_hz: tuple[float, float] = (2.5, 5.0)
    demodulation: str = "arctan"
    dc_compensation: str = "circle"
    min_arc_rad: float = 0.3
    max_circle_residual: float = 0.3
    lo_drift_compensation: bool = True
    bandpass_order: int = 4
    detection: DetectionSettings = field(default_factory=DetectionSettings)

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "VitalSignsSettings":
        """Build the settings from the ``vital_signs`` section (defaults for missing keys)."""
        defaults = cls()
        return cls(
            window_s=float(section.get("window_s", defaults.window_s)),
            update_period_s=float(section.get("update_period_s", defaults.update_period_s)),
            breathing_band_hz=tuple(section.get("breathing_band_hz", defaults.breathing_band_hz)),
            heart_band_hz=tuple(section.get("heart_band_hz", defaults.heart_band_hz)),
            reference_band_hz=tuple(section.get("reference_band_hz", defaults.reference_band_hz)),
            demodulation=str(section.get("demodulation", defaults.demodulation)),
            dc_compensation=str(section.get("dc_compensation", defaults.dc_compensation)),
            min_arc_rad=float(section.get("min_arc_rad", defaults.min_arc_rad)),
            max_circle_residual=float(section.get("max_circle_residual", defaults.max_circle_residual)),
            lo_drift_compensation=bool(section.get("lo_drift_compensation", defaults.lo_drift_compensation)),
            bandpass_order=int(section.get("bandpass_order", defaults.bandpass_order)),
            detection=DetectionSettings.from_config(section.get("detection", {})),
        )


@dataclass
class WindowAnalysis:
    """Everything computed on one analysis window.

    Displacements are in millimetres (arbitrary units when
    ``demodulation.calibrated`` is ``False``), positive away from the radar.

    Attributes
    ----------
    t_s : numpy.ndarray
        Time axis of the window (s, from its start).
    displacement_mm : numpy.ndarray
        Displacement, mean removed.
    breath_mm : numpy.ndarray
        Displacement band-passed in the breathing band.
    heart_mm : numpy.ndarray or None
        Displacement band-passed in the heartbeat band (``None`` if the
        slow-time rate is too low).
    spectrum_f_hz, spectrum_psd : numpy.ndarray
        One-sided periodogram of the detrended displacement.
    metrics : BreathMetrics
        Breathing metrics and instantaneous confidence of the window.
    cycles : estimation.BreathCycles
        Breath-by-breath analysis of ``breath_mm``.
    iq : numpy.ndarray
        Complex window actually demodulated (after the LO-drift derotation),
        for the IQ-constellation display.
    rates_hz : dict
        Cross-check estimators: ``fft``, ``acf``, ``peaks``, ``zero_crossing``
        (the detector uses ``metrics.rate_hz``).
    heart_rate_hz : float or None
        Heartbeat-band spectral peak (indicative only).
    lo_drift_hz : float
        Common rotation removed before the demodulation (Hz).
    demodulation : DemodulationResult
        Demodulation details (method used, circle fit).
    """

    t_s: np.ndarray
    displacement_mm: np.ndarray
    breath_mm: np.ndarray
    heart_mm: np.ndarray | None
    spectrum_f_hz: np.ndarray
    spectrum_psd: np.ndarray
    metrics: BreathMetrics
    cycles: estimation.BreathCycles
    iq: np.ndarray
    rates_hz: dict[str, float | None]
    heart_rate_hz: float | None
    lo_drift_hz: float
    demodulation: DemodulationResult


class VitalSignsProcessor:
    """Analyse windows of complex slow-time samples.

    Stateless between calls: each call re-analyses the latest window, which
    keeps the DC-offset estimate adapted to slow scene changes and allows
    zero-phase filters.

    Parameters
    ----------
    f_s_hz : float
        Slow-time sampling rate (Hz); must exceed ``2 × reference_band_hz[1]``.
    wavelength_m : float
        Carrier wavelength λ (m), for the phase → displacement conversion.
    settings : VitalSignsSettings
        Processing parameters.
    """

    def __init__(self, f_s_hz: float, wavelength_m: float, settings: VitalSignsSettings) -> None:
        self.f_s_hz = float(f_s_hz)
        self.wavelength_m = float(wavelength_m)
        self.settings = settings
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
        self._heart_band_usable = settings.heart_band_hz[1] < nyquist_hz
        self._n_fft = int(2 ** np.ceil(np.log2(self.window_length * _ZERO_PAD)))
        logger.info(
            "VitalSignsProcessor — %.2f Hz slow time, %d-sample window (%.1f s), "
            "λ=%.4f m, demodulation=%s, DC=%s",
            self.f_s_hz, self.window_length, settings.window_s, self.wavelength_m,
            settings.demodulation, settings.dc_compensation,
        )

    def _recent_peak_to_peak_mm(self, z: np.ndarray) -> float:
        """Peak-to-peak displacement (mm) over the last ``motion_window_s`` seconds.

        The short segment gets its own circle fit, so a large movement is
        measured in millimetres even when the full window cannot be (NaN if
        the short segment cannot be demodulated either).
        """
        n = int(round(self.settings.detection.motion_window_s * self.f_s_hz))
        if n < 4 or n >= z.size:
            return float("nan")
        result = demodulate(z[-n:], "arctan", "circle", self.settings.min_arc_rad,
                            self.settings.max_circle_residual)
        if not result.calibrated:
            return float("nan")
        return float(np.ptp(phase_to_displacement_m(result.phase_rad, self.wavelength_m)) * 1e3)

    def analyse(self, z: np.ndarray) -> WindowAnalysis:
        """Run the vital-sign analysis on one window of complex slow-time samples."""
        s = self.settings
        z = np.asarray(z, dtype=np.complex128)
        t_s = np.arange(z.size) / self.f_s_hz

        lo_drift_hz = 0.0
        if s.lo_drift_compensation:
            z, lo_drift_hz = remove_common_rotation(z, self.f_s_hz)
        result = demodulate(z, s.demodulation, s.dc_compensation, s.min_arc_rad, s.max_circle_residual)
        if result.calibrated:
            displacement_mm = phase_to_displacement_m(result.phase_rad, self.wavelength_m) * 1e3
        else:
            displacement_mm = result.phase_rad  # arbitrary units (linear fallback)
        displacement_mm = displacement_mm - displacement_mm.mean()
        # The band-pass filters remove the drift themselves; the linear
        # detrend is only used for the spectrum and the motion check, because
        # on a short window it also biases the breathing amplitude.
        detrended_mm = detrend(displacement_mm)

        breath_mm = bandpass(displacement_mm, self.f_s_hz,
                             widened_band(s.breathing_band_hz, self.f_s_hz), s.bandpass_order)
        heart_mm = None
        if self._heart_band_usable:
            heart_mm = bandpass(displacement_mm, self.f_s_hz,
                                widened_band(s.heart_band_hz, self.f_s_hz, 1.2), s.bandpass_order)

        spectrum = estimation.periodogram(detrended_mm, self.f_s_hz, n_fft=self._n_fft)
        # Largest peak-to-peak displacement of the whole window and of its
        # recent part (NaN when neither can be measured in millimetres).
        whole_window_ptp_mm = float(np.ptp(detrended_mm)) if result.calibrated else float("nan")
        measurable = [v for v in (whole_window_ptp_mm, self._recent_peak_to_peak_mm(z)) if np.isfinite(v)]
        displacement_ptp_mm = max(measurable) if measurable else float("nan")
        metrics = breathing_metrics(spectrum, displacement_ptp_mm, s.breathing_band_hz,
                                    s.reference_band_hz, s.detection)

        cycles = estimation.breath_cycles(breath_mm, self.f_s_hz, s.breathing_band_hz)
        rates_hz = {
            "fft": metrics.rate_hz,
            "acf": estimation.acf_rate(breath_mm, self.f_s_hz, s.breathing_band_hz)[0],
            "peaks": cycles.rate_hz,
            "zero_crossing": estimation.zero_crossing_rate(breath_mm, self.f_s_hz, s.breathing_band_hz),
        }
        heart_rate_hz = None
        if heart_mm is not None:
            heart_rate_hz = estimation.fft_peak_rate(heart_mm, self.f_s_hz, s.heart_band_hz)[0]

        logger.debug(
            "Window — rate=%s Hz, SNR=%.1f dB, concentration=%.2f, motion=%s, confidence=%.2f, demod=%s",
            f"{metrics.rate_hz:.3f}" if metrics.rate_hz else "n/a", metrics.snr_db,
            metrics.concentration, metrics.motion, metrics.confidence, result.method,
        )
        return WindowAnalysis(
            t_s=t_s,
            displacement_mm=displacement_mm,
            breath_mm=breath_mm,
            heart_mm=heart_mm,
            spectrum_f_hz=spectrum.f_hz,
            spectrum_psd=spectrum.psd,
            metrics=metrics,
            cycles=cycles,
            iq=z,
            rates_hz=rates_hz,
            heart_rate_hz=heart_rate_hz,
            lo_drift_hz=lo_drift_hz,
            demodulation=result,
        )


@dataclass
class PipelineOutput:
    """One result of :class:`VitalSignsPipeline` (one every ``update_period_s``).

    Attributes
    ----------
    update_index : int
        Number of the output since the start (1, 2, ...).
    t_s : float
        Slow time of the output (s since the start of the stream).
    discontinuity : bool
        ``True`` if the stream had a discontinuity since the previous output
        (the processing was then restarted).
    analysis : WindowAnalysis or None
        Analysis of the latest window, ``None`` while the window fills.
    breathing : BreathingState
        Decision of the detector (state ``WARMUP`` while the window fills).
    micro_doppler_column_db : numpy.ndarray or None
        Micro-Doppler spectrum of the latest slow-time segment (dB, centred,
        see :attr:`VitalSignsPipeline.micro_doppler_f_hz`), for display.
    """

    update_index: int
    t_s: float
    discontinuity: bool
    analysis: WindowAnalysis | None
    breathing: BreathingState
    micro_doppler_column_db: np.ndarray | None


class VitalSignsPipeline:
    """Phase-demodulation pipeline from IQ blocks to breathing decisions.

    Parameters
    ----------
    settings : VitalSignsSettings
        Window-analysis and detection parameters.
    input_rate_hz : float
        Sampling rate of the blocks (2 MS/s for the PlutoSDR, 2 kHz for the
        converted legacy recordings...).  ``input_rate_hz / slow_time_rate_hz``
        must be an integer made of prime factors ≤ 13.
    tx_offset_hz : float
        Frequency of the echo in the blocks (effective TX offset, 0 in pure CW).
    wavelength_m : float
        Carrier wavelength (m).
    slow_time_rate_hz : float, optional
        Rate of the slow-time signal after decimation (Hz).
    warmup_s : float, optional
        Slow time dropped after a (re)start, while the decimation filters settle (s).
    micro_doppler_window_s : float, optional
        Length of the slow-time segment of each micro-Doppler column (s).
    micro_doppler_n_fft : int, optional
        FFT size of the micro-Doppler columns (zero-padded).
    micro_doppler_window : str, optional
        Analysis window of the micro-Doppler columns.

    Notes
    -----
    **Discontinuities.**  A block with ``overflow`` set, or whose
    ``sample_start`` is not the end of the previous block, means that samples
    are missing: the phase of the echo is then unknown.  The pipeline resets
    its oscillator, filters, slow-time history and detector, marks the next
    output with ``discontinuity=True`` and starts filling the window again.
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
        self._mixer = Mixer(input_rate_hz, tx_offset_hz)
        max_frequency_hz = max(settings.reference_band_hz[1], settings.heart_band_hz[1])
        self._decimator = Decimator(input_rate_hz, int(round(decimation_ratio)), max_frequency_hz)
        self._processor = VitalSignsProcessor(slow_time_rate_hz, wavelength_m, settings)
        self._warmup_samples = int(round(warmup_s * slow_time_rate_hz))
        self._update_samples = max(1, int(round(settings.update_period_s * slow_time_rate_hz)))
        self._micro_doppler_samples = int(round(micro_doppler_window_s * slow_time_rate_hz))
        self._micro_doppler_window = get_window(micro_doppler_window, self._micro_doppler_samples)
        self._micro_doppler_n_fft = int(micro_doppler_n_fft)
        self.micro_doppler_f_hz = frequency_axis(self._micro_doppler_n_fft, self.slow_time_rate_hz)

        history_length = max(self._processor.window_length, self._micro_doppler_samples)
        self._history: deque[complex] = deque(maxlen=history_length)
        self._next_sample_start: int | None = None
        self._slow_samples_total = 0
        self._update_index = 0
        self._discontinuity_pending = False
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], input_rate_hz: float, tx_offset_hz: float) -> "VitalSignsPipeline":
        """Build the pipeline from the configuration (``sdr``, ``slow_time``,
        ``vital_signs`` and ``micro_doppler_view`` sections)."""
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
        )

    def reset(self) -> None:
        """Forget the past: restart the oscillator, filters, history and detector."""
        self._mixer.reset()
        self._decimator.reset()
        self._history.clear()
        self._samples_in_history = 0
        self._samples_since_update = 0
        self._warmup_remaining = self._warmup_samples
        self._detector = BreathingDetector(self.settings.detection, self.settings.update_period_s)

    def process_block(self, block: Block) -> list[PipelineOutput]:
        """Process one block (``samples``, ``sample_start``, ``overflow``); return the new outputs."""
        samples = block.samples[0]
        if self._next_sample_start is not None and (
            block.overflow or block.sample_start != self._next_sample_start
        ):
            logger.warning("Stream discontinuity before sample %d — processing restarted", block.sample_start)
            self.reset()
            self._discontinuity_pending = True
        self._next_sample_start = block.sample_start + samples.size

        slow_samples = self._decimator(self._mixer(samples))
        outputs: list[PipelineOutput] = []
        for value in slow_samples:
            self._slow_samples_total += 1
            if self._warmup_remaining > 0:  # decimation filters still settling
                self._warmup_remaining -= 1
                continue
            self._history.append(complex(value))
            self._samples_in_history += 1
            self._samples_since_update += 1
            if self._samples_since_update == self._update_samples:
                self._samples_since_update = 0
                outputs.append(self._make_output())
        return outputs

    def run(self, source: Source) -> Iterator[PipelineOutput]:
        """Process every block of *source* (and close it at the end)."""
        try:
            while True:
                block = source.read_block()
                if block is None:
                    break
                yield from self.process_block(block)
        finally:
            source.close()

    def _make_output(self) -> PipelineOutput:
        """Analyse the latest window and build one output."""
        t_s = self._slow_samples_total / self.slow_time_rate_hz
        history = np.array(self._history, dtype=np.complex128)

        micro_doppler_column_db = None
        if self._samples_in_history >= self._micro_doppler_samples:
            segment = history[-self._micro_doppler_samples:]
            # The segment mean is the static-clutter line at 0 Hz; removing it
            # is fine for this magnitude display (never before the phase).
            column = compute_single_column(segment - segment.mean(), self.slow_time_rate_hz,
                                           self._micro_doppler_window, self._micro_doppler_n_fft)
            micro_doppler_column_db = column.power_db

        analysis = None
        breathing = self._detector.warmup(t_s)
        if self._samples_in_history >= self._processor.window_length:
            analysis = self._processor.analyse(history[-self._processor.window_length:])
            breathing = self._detector.update(analysis.metrics, t_s, analysis.cycles.since_last_s)

        self._update_index += 1
        output = PipelineOutput(self._update_index, t_s, self._discontinuity_pending,
                                analysis, breathing, micro_doppler_column_db)
        self._discontinuity_pending = False
        return output
