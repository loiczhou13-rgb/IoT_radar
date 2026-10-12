#!/usr/bin/env python3
"""Run the radar live: acquisition → phase-demodulation pipeline → dashboard.

Usage (from the repository root, after ``pip install -e .``)::

    python scripts/run_radar.py                 # PlutoSDR, with the dashboard
    python scripts/run_radar.py --simulation    # no hardware
    python scripts/run_radar.py --headless      # no window: results in the log
    python scripts/run_radar.py --measure       # one measurement over a silence (M4), then exit
    python scripts/run_radar.py --config my_setup.yaml

With the dashboard, the key ``m`` starts a measurement (M4): stand still,
out of the beam, for ``measurement.duration_s`` (30 s by default).
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
from typing import Any, Callable, Iterator

from iot_radar.acquisition.pluto import ADC_FULL_SCALE, effective_tx_offset_hz
from iot_radar.acquisition.sources import Source, open_source
from iot_radar.config import DEFAULT_RADAR_CONFIG, load_config, radar_log_file, setup_logging
from iot_radar.measurement import MeasurementResult
from iot_radar.physics import SPEED_OF_LIGHT, range_interval_m
from iot_radar.pipeline import PipelineOutput, VitalSignsPipeline

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="CW radar — breathing detection of buried people")
    parser.add_argument("--config", default=str(DEFAULT_RADAR_CONFIG),
                        help="YAML configuration file (default: %(default)s)")
    parser.add_argument("--simulation", action="store_true",
                        help="use the simulated source (no PlutoSDR needed)")
    parser.add_argument("--headless", action="store_true",
                        help="no dashboard: write every result to the log")
    parser.add_argument("--measure", action="store_true",
                        help="headless: one measurement over a silence (M4) from the start, then exit")
    parser.add_argument("--log-file", default=None,
                        help="explicit log file (default: logs/radar_<timestamp>.log)")
    return parser.parse_args(argv)


def log_output(output: PipelineOutput) -> None:
    """One log line per pipeline output."""
    breathing = output.breathing
    rate = f"{breathing.rate_bpm:.1f} /min" if breathing.rate_bpm else "—"
    extra = ""
    if breathing.pause_reason:
        extra += f"  pause={breathing.pause_reason}"
    if output.contamination is not None:
        extra += (f"  weight={output.contamination.weight:.2f}"
                  f"  contaminated={100 * output.contamination.contaminated_fraction:.0f}%")
    if output.measurement is not None and output.measurement.state == "running":
        extra += f"  measuring {output.measurement.elapsed_s:.0f}/{output.measurement.duration_s:.0f} s"
    logger.info(
        "t=%6.1f s  %-12s confidence=%.2f  rate=%s%s%s%s",
        output.t_s, breathing.state, breathing.confidence, rate, extra,
        "  APNEA?" if breathing.apnea else "",
        "  (stream discontinuity: restarted)" if output.discontinuity else "",
    )


def log_measurement(result: MeasurementResult) -> None:
    """Summary of a measurement (M4) in the log."""
    rate = f"{result.rate_bpm:.1f} /min" if result.rate_bpm else "—"
    logger.info(
        "MEASUREMENT %.1f–%.1f s: %s%s  rate=%s  confidence=%.2f  contaminated=%.0f%%  saturated=%.0f%%  "
        "activity around=%.0f%%  target motion=%s",
        result.t_start_s, result.t_end_s, result.decision.upper(),
        f" ({', '.join(result.reasons)})" if result.reasons else "", rate, result.confidence,
        100 * result.contaminated_fraction, 100 * result.saturated_fraction, 100 * result.activity_fraction,
        "yes" if result.target_motion else "no",
    )


def radar_outputs(pipeline: VitalSignsPipeline, source: Source,
                  measure_request: threading.Event | None = None) -> Iterator[PipelineOutput]:
    """Outputs of *pipeline* on *source* (closed at the end).

    A measurement starts when *measure_request* is set (key ``m`` of the
    dashboard, GUI thread): the request is read here, before each block, so
    the pipeline is only used by the producer thread.
    """
    try:
        while True:
            if measure_request is not None and measure_request.is_set():
                measure_request.clear()
                pipeline.start_measurement()
            block = source.read_block()
            if block is None:
                break
            for output in pipeline.process_block(block):
                if output.measurement is not None and output.measurement.state == "done":
                    log_measurement(output.measurement.result)
                yield output
    finally:
        source.close()


def dashboard_context(
    cfg: dict[str, Any],
    pipeline: VitalSignsPipeline,
    tx_offset_hz: float,
    title: str,
    source_name: str,
) -> dict[str, Any]:
    """Static description of the run shown by :class:`iot_radar.ui.dashboard.PhaseDashboard`."""
    f_c_hz = float(cfg["sdr"]["center_frequency_hz"])
    wavelength_m = SPEED_OF_LIGHT / f_c_hz
    range_min_m, range_max_m = range_interval_m(cfg)
    settings = pipeline.settings
    detection = settings.detection
    breathing_band_hz, heart_band_hz = settings.breathing_band_hz, settings.heart_band_hz
    display = cfg.get("display", {})
    radar_rows = [
        ("Source", source_name),
        ("Carrier f_c", f"{f_c_hz / 1e9:.3f} GHz"),
        ("Wavelength λ", f"{wavelength_m * 100:.2f} cm"),
        ("Phase sensitivity", f"{4 * math.pi / wavelength_m:.0f} rad/m"),
        ("TX offset", f"{tx_offset_hz:.2f} Hz"),
        ("Input rate", f"{pipeline.input_rate_hz / 1e3:.1f} kS/s"),
        ("Slow-time rate", f"{pipeline.slow_time_rate_hz:.0f} Hz"),
        ("Link-budget range", f"{range_min_m:.0f}–{range_max_m:.0f} m"),
    ]
    processing_rows = [
        ("Analysis window", f"{settings.window_s:.0f} s, every {settings.update_period_s} s"),
        ("Breathing band", f"{breathing_band_hz[0]}–{breathing_band_hz[1]} Hz"),
        ("", f"{60 * breathing_band_hz[0]:.0f}–{60 * breathing_band_hz[1]:.0f} /min"),
        ("Heart band", f"{heart_band_hz[0]}–{heart_band_hz[1]} Hz"),
        ("Noise ref. band", f"{settings.reference_band_hz[0]}–{settings.reference_band_hz[1]} Hz"),
        ("DC compensation", settings.dc_compensation),
        ("Demodulation", settings.demodulation),
        ("Circle fit", f"arc ≥ {settings.min_arc_rad} rad"),
        ("", f"residual ≤ {settings.max_circle_residual}"),
    ]
    detection_rows = [
        ("SNR score 0→1", f"{detection.snr_min_db:.0f} → {detection.snr_good_db:.0f} dB"),
        ("Conc. score 0→1", f"{detection.concentration_floor} → {detection.concentration_good}"),
        ("Peak window", f"±{detection.peak_halfwidth_hz} Hz"),
        ("Threshold ON/OFF", f"{detection.threshold_on} / {detection.threshold_off}"),
        ("Smoothing τ", f"{detection.smoothing_tau_s:.0f} s"),
        ("Motion limit", f"{detection.motion_ptp_mm:.0f} mm p-p"),
        ("Motion hold", f"{detection.motion_hold_s:.0f} s"),
        ("Apnea alert", f"no breath ≥ {detection.apnea_s:.0f} s"),
    ]
    motion = pipeline.motion
    robustness_rows = [
        ("M1 Doppler filter", f"{motion.cutoff_hz} Hz, delay {pipeline.doppler_delay_s:.2f} s"
         if motion.doppler_filter_enabled else "off"),
        ("Aliasing into ±1 Hz", f"≥ {pipeline.alias_attenuation_db:.0f} dB"),
        ("M2 centre tracking", f"{motion.center_method} {motion.center_window_s:.0f} s / {motion.center_step_s:.0f} s"
         if motion.center_tracking_enabled else "off"),
        ("M3 weighting", f"κ₀ {motion.weight_midpoint_db:.0f} dB, slope {motion.weight_slope_db:.0f} dB"
         if motion.contamination_enabled else "off (former pause)"),
        ("M4 measurement", f"{pipeline.measurement_settings.duration_s:.0f} s (key m)"),
    ]
    return {
        "title": title,
        "subtitle": f"{source_name} · f_c {f_c_hz / 1e9:.2f} GHz · window {settings.window_s:.0f} s",
        "micro_doppler_f_hz": pipeline.micro_doppler_f_hz,
        "display_max_hz": float(cfg.get("micro_doppler_view", {}).get("display_max_hz", 3.0)),
        "waterfall_columns": int(display.get("waterfall_columns", 240)),
        "history_s": float(display.get("history_s", 120.0)),
        "breathing_band_hz": breathing_band_hz,
        "heart_band_hz": heart_band_hz,
        "reference_band_hz": settings.reference_band_hz,
        "threshold_on": detection.threshold_on,
        "threshold_off": detection.threshold_off,
        "peak_halfwidth_hz": detection.peak_halfwidth_hz,
        "parameters": [("Radar", radar_rows), ("Processing", processing_rows), ("Detection", detection_rows),
                       ("Motion robustness", robustness_rows)],
    }


def show_or_log(outputs, headless: bool, context: dict[str, Any], full_screen: bool,
                on_measure: Callable[[], None] | None = None) -> None:
    """Display the outputs in the dashboard, or log them when *headless*.

    *on_measure* is called when the key ``m`` of the dashboard is pressed.
    """
    if headless:
        try:
            for output in outputs:
                log_output(output)
        except KeyboardInterrupt:
            logger.info("Stopped by the user")
        return
    from iot_radar.ui.dashboard import PhaseDashboard  # matplotlib window only when needed

    PhaseDashboard(context, full_screen=full_screen, on_measure=on_measure).run(outputs)


def main(argv: list[str] | None = None) -> None:
    """Open the source, run the pipeline and show its results until stopped."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"), radar_log_file(cfg, args.log_file))
    logger.info("=== Radar started — configuration %s ===", args.config)

    source = open_source(cfg, simulation=args.simulation)
    tx_offset_hz = effective_tx_offset_hz(cfg)
    # Pluto and simulation deliver raw ADC values: M3 pauses on saturation.
    pipeline = VitalSignsPipeline.from_config(cfg, source.sample_rate_hz, tx_offset_hz,
                                              adc_full_scale=ADC_FULL_SCALE)
    if args.measure:
        run_one_measurement(pipeline, source)
        logger.info("=== Radar stopped ===")
        return
    source_name = "simulation" if source.kind == "simulation" else f"PlutoSDR {cfg['sdr']['uri']}"
    context = dashboard_context(cfg, pipeline, tx_offset_hz, "CW radar — breathing detection", source_name)
    measure_request = threading.Event()
    show_or_log(radar_outputs(pipeline, source, measure_request), args.headless, context,
                bool(cfg.get("display", {}).get("full_screen", False)), on_measure=measure_request.set)
    logger.info("=== Radar stopped ===")


def run_one_measurement(pipeline: VitalSignsPipeline, source: Source) -> MeasurementResult | None:
    """Headless M4: measure from now on (the silence must start now), log the result and return it."""
    logger.info("Measurement of %.0f s from now: stay still, out of the beam",
                pipeline.measurement_settings.duration_s)
    pipeline.start_measurement()
    outputs = radar_outputs(pipeline, source)
    try:
        for output in outputs:
            log_output(output)
            if output.measurement is not None and output.measurement.state == "done":
                return output.measurement.result
    except KeyboardInterrupt:
        logger.info("Stopped by the user")
    finally:
        outputs.close()
    return None


if __name__ == "__main__":
    main()
