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
from iot_radar.pipeline import build_context, streaming_frame_generator
from iot_radar.ui.dashboard import DashboardRadar

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Radar micro-Doppler — détection de survivants ensevelis",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_RADAR_CONFIG),
        help="Chemin vers le fichier de configuration YAML (défaut : %(default)s)",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Forcer le mode simulation (pas de PlutoSDR requis)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Chemin explicite du fichier de log.  Par défaut, "
            "logs/radar_<timestamp>.log."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Top-level pipeline orchestration (streaming mode)."""
    args = parse_args()
    cfg = load_config(args.config)
    log_path = radar_log_file(cfg, args.log_file)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"), log_path)

    logger.info("=== Démarrage du pipeline micro-Doppler (mode continu) ===")
    logger.info("Configuration chargée depuis %s", args.config)
    if log_path is not None:
        logger.info("Logs persistants : %s", log_path)

    context = build_context(cfg)
    dashboard = DashboardRadar(config=cfg, context=context)
    gen = streaming_frame_generator(cfg, simulation=args.simulation)
    dashboard.run(gen)

    logger.info("=== Pipeline terminé ===")


if __name__ == "__main__":
    main()
