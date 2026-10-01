#!/usr/bin/env python3
"""Replay a recorded HDF5 session through the phase-demodulation pipeline and the dashboard.

The raw IQ blocks of the session are fed to the same pipeline as a live run
(:class:`iot_radar.acquisition.sources.ReplaySource`), so the replay shows
what the radar computed — or what it would compute with another
configuration (``--config``).

Examples (from the repository root)::

    python scripts/replay.py data/sessions/session_0001_20261005_143012.h5
    python scripts/replay.py SESSION.h5 --speed 4      # four times faster
    python scripts/replay.py SESSION.h5 --speed 0      # as fast as possible
    python scripts/replay.py SESSION.h5 --headless     # no window: results in the log
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from iot_radar.acquisition.sources import ReplaySource
from iot_radar.config import DEFAULT_RADAR_CONFIG, load_config, radar_log_file, setup_logging
from iot_radar.pipeline import VitalSignsPipeline
from run_radar import dashboard_context, show_or_log

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options of the replay."""
    p = argparse.ArgumentParser(description="Replay a recorded HDF5 session through the radar pipeline.")
    p.add_argument("session", type=Path, help="session file (.h5)")
    p.add_argument("--config", default=None,
                   help="YAML configuration (default: the one stored in the session)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="replay speed: 1 = real time (default), 2 = twice faster, 0 = as fast as possible")
    p.add_argument("--headless", action="store_true", help="no dashboard: write every result to the log")
    p.add_argument("--log-file", default=None, help="log file (default: from the configuration)")
    return p.parse_args(argv)


def replay_config(source: ReplaySource, config_path: str | None) -> dict[str, Any]:
    """Configuration of the replay: ``--config``, else the one stored in the session.

    Sessions converted from legacy recordings store no configuration; the
    default configuration is then used, with a warning.
    """
    if config_path is not None:
        return load_config(config_path)
    stored = source.attributes.get("config_yaml", "")
    if stored:
        return yaml.safe_load(stored)
    logger.warning("No configuration stored in the session — using %s", DEFAULT_RADAR_CONFIG)
    return load_config(DEFAULT_RADAR_CONFIG)


def session_config(cfg: dict[str, Any], source: ReplaySource) -> dict[str, Any]:
    """*cfg* with the carrier frequency of the session (the wavelength depends on it)."""
    return dict(cfg, sdr={**cfg["sdr"], "center_frequency_hz": float(source.attributes["center_frequency_hz"])})


def build_pipeline(cfg: dict[str, Any], source: ReplaySource) -> VitalSignsPipeline:
    """Pipeline for the session: its sampling rate and TX offset come from the file."""
    return VitalSignsPipeline.from_config(cfg, source.sample_rate_hz, float(source.attributes["tx_offset_hz"]))


def main(argv: list[str] | None = None) -> None:
    """Open the session and run the pipeline on it."""
    args = parse_args(argv)
    if not args.session.is_file():
        raise SystemExit(f"File not found: {args.session}")
    source = ReplaySource(args.session, speed=args.speed or None)
    cfg = session_config(replay_config(source, args.config), source)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"), radar_log_file(cfg, args.log_file))
    labels = sorted({label for _, _, label in source.annotations()})
    logger.info("Replaying %s (annotated: %s)", args.session.name, ", ".join(labels) or "none")
    pipeline = build_pipeline(cfg, source)
    title = f"Replay — {args.session.name}" + (f" — {', '.join(labels)}" if labels else "")
    context = dashboard_context(cfg, pipeline, float(source.attributes["tx_offset_hz"]), title,
                                f"replay of a {source.attributes.get('source_kind', '?')} session")
    show_or_log(pipeline.run(source), args.headless, context,
                bool(cfg.get("display", {}).get("full_screen", False)))


if __name__ == "__main__":
    main()
