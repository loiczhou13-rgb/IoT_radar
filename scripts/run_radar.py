#!/usr/bin/env python3
"""Run the radar live: acquisition → phase-demodulation pipeline → results.

Usage (from the repository root, after ``pip install -e .``)::

    python scripts/run_radar.py                 # PlutoSDR
    python scripts/run_radar.py --simulation    # no hardware
    python scripts/run_radar.py --config my_setup.yaml

Every result (state, confidence, breathing rate) is written to the log.
"""

from __future__ import annotations

import argparse
import logging

from iot_radar.acquisition.pluto import effective_tx_offset_hz
from iot_radar.acquisition.sources import open_source
from iot_radar.config import DEFAULT_RADAR_CONFIG, load_config, radar_log_file, setup_logging
from iot_radar.pipeline import PipelineOutput, VitalSignsPipeline

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="CW radar — breathing detection of buried people")
    parser.add_argument("--config", default=str(DEFAULT_RADAR_CONFIG),
                        help="YAML configuration file (default: %(default)s)")
    parser.add_argument("--simulation", action="store_true",
                        help="use the simulated source (no PlutoSDR needed)")
    parser.add_argument("--log-file", default=None,
                        help="explicit log file (default: logs/radar_<timestamp>.log)")
    return parser.parse_args(argv)


def log_output(output: PipelineOutput) -> None:
    """One log line per pipeline output."""
    breathing = output.breathing
    rate = f"{breathing.rate_bpm:.1f} /min" if breathing.rate_bpm else "—"
    logger.info(
        "t=%6.1f s  %-12s confidence=%.2f  rate=%s%s%s",
        output.t_s, breathing.state, breathing.confidence, rate,
        "  APNEA?" if breathing.apnea else "",
        "  (stream discontinuity: restarted)" if output.discontinuity else "",
    )


def main(argv: list[str] | None = None) -> None:
    """Open the source, run the pipeline until it stops (Ctrl-C)."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"), radar_log_file(cfg, args.log_file))
    logger.info("=== Radar started — configuration %s ===", args.config)

    source = open_source(cfg, simulation=args.simulation)
    pipeline = VitalSignsPipeline.from_config(cfg, source.sample_rate_hz, effective_tx_offset_hz(cfg))
    try:
        for output in pipeline.run(source):
            log_output(output)
    except KeyboardInterrupt:
        logger.info("Stopped by the user")
    logger.info("=== Radar stopped ===")


if __name__ == "__main__":
    main()
