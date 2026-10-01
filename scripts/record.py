#!/usr/bin/env python3
"""Record radar sessions (raw IQ) into HDF5 files, without display.

Examples (from the repository root) — a 2-minute session with a breathing
person 2.5 m behind 20 cm of concrete::

    python scripts/record.py --label breathing --duration-s 120 \
        --subject-id S01 --distance-m 2.5 --obstacle concrete \
        --obstacle-thickness-cm 20 --room lab_b12

Five 60 s sessions of an empty room with a 2-minute pause after each one::

    python scripts/record.py -n 5 --interval-s 120 --label empty --duration-s 60 --room lab_b12

Simulated session (the label then sets the simulated scene; without
``--label`` the scene of the configuration is used)::

    python scripts/record.py --simulation --duration-s 30

Each session is written to ``<sessions_dir>/session_<id>_<YYYYMMDD>_<HHMMSS>.h5``
(UTC start time, see ``iot_radar.acquisition.recording`` for the layout).  The
whole session is annotated with ``--label``; a simulated session also stores
the simulated chest displacement as ground truth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from iot_radar.acquisition.recording import (
    LABELS,
    SessionWriter,
    next_session_id,
    radar_attributes,
    session_file_name,
)
from iot_radar.acquisition.sources import CWSimulationSource, open_source
from iot_radar.config import (
    DEFAULT_RADAR_CONFIG,
    load_config,
    radar_log_file,
    resolve_repo_path,
    setup_logging,
)

logger = logging.getLogger(__name__)

GROUND_TRUTH_RATE_HZ: float = 100.0
"""Sampling rate of the simulated ground truth stored in the sessions (Hz)."""

SIMULATED_LABELS: tuple[str, ...] = ("breathing", "empty")
"""Labels the simulated scene can reproduce (presence or not of a breathing person)."""


def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line options of the recording."""
    p = argparse.ArgumentParser(description="Record radar sessions (raw IQ) into HDF5 files.")
    p.add_argument("--label", choices=LABELS, default=None,
                   help="annotation of the whole session (required with the PlutoSDR)")
    p.add_argument("--duration-s", type=float, default=120.0,
                   help="duration of each session in seconds of signal (default: 120)")
    p.add_argument("--config", default=str(DEFAULT_RADAR_CONFIG), help="YAML configuration file")
    p.add_argument("--simulation", action="store_true", help="record the simulated source (no PlutoSDR)")
    p.add_argument("--sessions-dir", type=Path, default=None,
                   help="output folder (default: recording.sessions_dir of the configuration)")
    p.add_argument("--session-id", type=int, default=None,
                   help="identifier of the session (default: next free one)")
    p.add_argument("--log-file", default=None, help="log file (default: from the configuration)")

    scene = p.add_argument_group("scene (stored as metadata)")
    scene.add_argument("--subject-id", default="", help="pseudonymous identifier of the person")
    scene.add_argument("--distance-m", type=float, default=float("nan"), help="radar-to-person distance (m)")
    scene.add_argument("--orientation", default="", help="e.g. front, back, side")
    scene.add_argument("--obstacle", default="", help="material between the radar and the person")
    scene.add_argument("--obstacle-thickness-cm", type=float, default=float("nan"),
                       help="thickness of the obstacle (cm)")
    scene.add_argument("--room", default="", help="place of the measurement")
    scene.add_argument("--notes", default="", help="free text")

    repetition = p.add_argument_group("repetition")
    repetition.add_argument("--samples", "-n", type=int, default=1, metavar="N",
                            help="number of successive sessions (default: 1)")
    repetition.add_argument("--interval-s", type=float, default=0.0,
                            help="pause after each session before the next one (s, default: 0)")
    return p


def scene_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Scene attributes given on the command line."""
    return {
        "subject_id": args.subject_id,
        "distance_m": args.distance_m,
        "orientation": args.orientation,
        "obstacle": args.obstacle,
        "obstacle_thickness_cm": args.obstacle_thickness_cm,
        "room": args.room,
        "notes": args.notes,
    }


def record_session(cfg: dict[str, Any], args: argparse.Namespace, session_id: int) -> Path:
    """Record one session and return the path of its file.

    The acquisition stops after ``args.duration_s`` seconds of **signal**
    (whole blocks), on Ctrl-C, or when the PlutoSDR link is lost; the file is
    closed properly in every case.
    """
    simulation = args.simulation or cfg["simulation"].get("enabled", False)
    source = open_source(cfg, simulation=args.simulation)
    radar = radar_attributes(cfg, source)
    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else resolve_repo_path(
        cfg["recording"]["sessions_dir"])
    start_time_utc = dt.datetime.now(dt.timezone.utc)
    path = sessions_dir / session_file_name(session_id, start_time_utc)
    n_samples_wanted = int(round(args.duration_s * source.sample_rate_hz))

    writer = SessionWriter(
        path,
        radar,
        scene=scene_from_args(args),
        config_yaml=yaml.safe_dump(cfg, sort_keys=False),
        start_time_utc=start_time_utc,
        with_ground_truth=simulation,
        flush_interval_s=float(cfg["recording"].get("flush_interval_s", 1.0)),
    )
    logger.info("Recording %.1f s of signal into %s", args.duration_s, path)
    try:
        while writer.n_samples < n_samples_wanted:
            block = source.read_block()
            if block is None:
                break
            writer.write_block(block.samples, block.host_time_s, block.overflow)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt — the session is saved as is.")
    except OSError as exc:
        link_lost = isinstance(exc, BrokenPipeError) or getattr(exc, "errno", None) in (
            errno.EPIPE, errno.ECONNRESET, errno.ETIMEDOUT, errno.ENOTCONN,
        )
        if not link_lost:
            raise
        logger.error(
            "PlutoSDR / libiio link lost (%s). Check the USB cable or the power "
            "supply, avoid weak hubs, disable USB sleep, check the IP address "
            "(sdr.uri) and that no other program uses the Pluto. The session is "
            "saved as is.", exc,
        )
    finally:
        label = args.label or (source.scene_label if isinstance(source, CWSimulationSource) else "unknown")
        writer.add_annotation(0, writer.n_samples, label)
        if isinstance(source, CWSimulationSource):
            time_s = np.arange(0.0, writer.n_samples / source.sample_rate_hz, 1.0 / GROUND_TRUTH_RATE_HZ)
            writer.set_ground_truth(time_s, source.chest_displacement_m(time_s))
        writer.close()
        source.close()
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    """Record ``--samples`` sessions (one by default)."""
    args = build_argument_parser().parse_args(argv)
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be > 0.")
    if args.samples < 1 or args.interval_s < 0:
        raise SystemExit("--samples must be >= 1 and --interval-s >= 0.")
    if args.samples > 1 and args.session_id is not None:
        raise SystemExit("--session-id is incompatible with -n > 1 (identifiers are automatic).")

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"), radar_log_file(cfg, args.log_file))
    simulation = args.simulation or cfg["simulation"].get("enabled", False)
    if args.label is None and not simulation:
        raise SystemExit("--label is required when recording the PlutoSDR.")
    if simulation and args.label is not None:
        if args.label not in SIMULATED_LABELS:
            raise SystemExit(f"The simulation can only reproduce the labels {SIMULATED_LABELS}.")
        cfg["simulation"]["presence"] = args.label == "breathing"

    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else resolve_repo_path(
        cfg["recording"]["sessions_dir"])
    paths: list[Path] = []
    for k in range(args.samples):
        if k > 0 and args.interval_s > 0:
            logger.info("Pause %.1f s before session %d / %d", args.interval_s, k + 1, args.samples)
            time.sleep(args.interval_s)
        session_id = args.session_id if args.session_id is not None else next_session_id(sessions_dir)
        paths.append(record_session(cfg, args, session_id))
    logger.info("Done — %d session(s): %s", len(paths), ", ".join(str(p) for p in paths))
    return paths


if __name__ == "__main__":
    main()
