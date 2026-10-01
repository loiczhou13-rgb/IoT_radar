#!/usr/bin/env python3
"""Run the radar pipeline live, with the dashboard.

Usage (from the repository root, after ``pip install -e .``)::

    python scripts/run_radar.py                 # PlutoSDR
    python scripts/run_radar.py --simulation    # no hardware
    python scripts/run_radar.py --config my_setup.yaml
"""

from __future__ import annotations

import argparse
import logging

from iot_radar.config import DEFAULT_RADAR_CONFIG, load_config, radar_log_file, setup_logging
from iot_radar.acquisition.sources import open_source
from iot_radar.pipeline import build_context, streaming_frame_generator
from iot_radar.ui.dashboard import DashboardRadar

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CW radar — breathing detection of buried people",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_RADAR_CONFIG),
        help="YAML configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use the simulated source (no PlutoSDR needed)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Explicit log file.  Default: "
            "logs/radar_<timestamp>.log."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Open the source, run the pipeline and show its frames until the window is closed."""
    args = parse_args()
    cfg = load_config(args.config)
    log_path = radar_log_file(cfg, args.log_file)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"), log_path)

    logger.info("=== Micro-Doppler pipeline started (streaming) ===")
    logger.info("Configuration loaded from %s", args.config)
    if log_path is not None:
        logger.info("Log file: %s", log_path)

    context = build_context(cfg)
    dashboard = DashboardRadar(config=cfg, context=context)
    source = open_source(cfg, simulation=args.simulation)
    frames = streaming_frame_generator(cfg, source)
    dashboard.run(frames)

    logger.info("=== Pipeline stopped ===")


if __name__ == "__main__":
    main()
