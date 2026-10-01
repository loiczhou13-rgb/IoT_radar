#!/usr/bin/env python3
"""Convert the legacy ``.iq`` recordings into HDF5 sessions (schema v1).

The former recording script wrote, next to each ``<n>.npz``, a ``<n>.iq``
file (complex64, decimated to ``f_s_dec_hz`` = 2 kHz and already filtered by
the clutter high-pass) and a ``<n>.json`` file of metadata.  This script turns
every such pair into a session file, so that the real recordings can be
replayed and re-processed by the current pipeline:

    python scripts/convert_legacy_iq.py AICalibration/data
    python scripts/convert_legacy_iq.py OLD_DIR --output-dir data/sessions/legacy

What is stored (see ``iot_radar.acquisition.recording``):

* ``/iq`` — int16 with ``scale = max(|I|, |Q|) / 30000`` (the samples are
  floats, not ADC integers); quantisation noise ~30 dB below the noise floor;
* ``sample_rate_hz`` from the ``.json``, ``iq_stage = "decimated_clutter_filtered"``,
  ``source_kind = "legacy_conversion"``;
* ``tx_offset_hz`` = 488.28125 Hz: the line actually transmitted by the old
  code (500 Hz requested, bug B1), unless ``--tx-offset-hz`` is given;
* carrier and gains: the values of the old default configuration, unless
  given on the command line (the old YAML files no longer exist);
* one ``/blocks`` row (the old files kept no block boundaries) and one
  annotation for the whole session (label 0 -> ``empty``, 1 -> ``breathing``);
* ``room`` = the old ``env``; ``legacy_source_file`` = the ``.iq`` path.

Files already converted (same ``legacy_source_file`` in the output folder)
are skipped.  The original files are only read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from iot_radar.acquisition.pluto import snap_tx_offset_hz
from iot_radar.acquisition.recording import (
    SessionReader,
    SessionWriter,
    next_session_id,
    session_file_name,
)
from iot_radar.config import resolve_repo_path, setup_logging

logger = logging.getLogger(__name__)

LEGACY_TX_OFFSET_HZ: float = snap_tx_offset_hz(500.0, 2e6, 16384)
"""Line actually emitted by the old configuration (500 Hz requested, see bug B1)."""

LEGACY_LABELS: dict[int, str] = {0: "empty", 1: "breathing"}
"""Old integer labels -> annotation labels."""

STORED_PEAK: float = 30000.0
"""int16 value given to the largest |I| or |Q| of a converted file."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options of the conversion."""
    p = argparse.ArgumentParser(description="Convert legacy .iq recordings into HDF5 sessions.")
    p.add_argument("inputs", nargs="+", type=Path, help=".iq files or folders searched recursively")
    p.add_argument("--output-dir", type=Path, default=Path("data/sessions/legacy"),
                   help="folder of the sessions (default: data/sessions/legacy)")
    p.add_argument("--center-frequency-hz", type=float, default=3.5e9)
    p.add_argument("--tx-offset-hz", type=float, default=LEGACY_TX_OFFSET_HZ)
    p.add_argument("--tx-gain-db", type=float, default=-20.0)
    p.add_argument("--rx-gain-db", type=float, default=45.0)
    return p.parse_args(argv)


def find_iq_files(inputs: list[Path]) -> list[Path]:
    """Every ``.iq`` file given directly or found under the given folders."""
    files: list[Path] = []
    for item in inputs:
        files += sorted(item.rglob("*.iq")) if item.is_dir() else [item]
    return files


def legacy_start_time(meta: dict[str, Any]) -> dt.datetime:
    """Start of the old recording: end time minus the recorded duration (UTC)."""
    end = dt.datetime.fromisoformat(meta["utc_finished"])
    return end - dt.timedelta(seconds=float(meta.get("duration_recorded_wall_s", 0.0)))


def convert_file(iq_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    """Convert one ``.iq`` + ``.json`` pair; return the session path."""
    meta = json.loads(iq_path.with_suffix(".json").read_text(encoding="utf-8"))
    iq = np.fromfile(iq_path, dtype=np.complex64)[np.newaxis, :]
    peak = float(max(np.abs(iq.real).max(initial=0.0), np.abs(iq.imag).max(initial=0.0)))
    iq_scale = peak / STORED_PEAK if peak > 0 else 1.0

    radar = {
        "sample_rate_hz": float(meta["f_s_dec_hz"]),
        "center_frequency_hz": args.center_frequency_hz,
        "tx_waveform": "cw_offset",
        "tx_offset_hz": args.tx_offset_hz,
        "tx_gain_db": args.tx_gain_db,
        "rx_gain_db": args.rx_gain_db,
        "rx_gain_mode": "manual",
        "n_rx_channels": 1,
        "channel_layout": "rx0",
        "firmware_version": "",
        "source_kind": "legacy_conversion",
        "iq_stage": "decimated_clutter_filtered",
    }
    scene = {
        "room": str(meta.get("env", "")),
        "notes": (f"converted from a legacy recording (subset {meta.get('subset', '?')}, "
                  f"sample {meta.get('sample_index', '?')}); carrier and gains assumed"),
    }
    start_time_utc = legacy_start_time(meta)
    path = output_dir / session_file_name(next_session_id(output_dir), start_time_utc)
    with SessionWriter(path, radar, scene=scene, start_time_utc=start_time_utc, iq_scale=iq_scale,
                       extra_attributes={"legacy_source_file": str(iq_path.resolve())}) as writer:
        writer.write_block(iq, host_time_s=0.0, overflow=False)
        writer.add_annotation(0, iq.shape[1], LEGACY_LABELS.get(int(meta.get("label", -1)), "unknown"))
    return path


def already_converted(output_dir: Path) -> set[str]:
    """``legacy_source_file`` of the sessions already present in *output_dir*."""
    sources: set[str] = set()
    for path in sorted(output_dir.glob("session_*.h5")) if output_dir.is_dir() else []:
        with SessionReader(path) as session:
            sources.add(str(session.attributes.get("legacy_source_file", "")))
    return sources


def main(argv: list[str] | None = None) -> list[Path]:
    """Convert every legacy recording found in the inputs."""
    args = parse_args(argv)
    setup_logging("INFO")
    output_dir = resolve_repo_path(args.output_dir)
    done = already_converted(output_dir)
    converted: list[Path] = []
    for iq_path in find_iq_files(args.inputs):
        if str(iq_path.resolve()) in done:
            logger.info("Already converted: %s", iq_path)
            continue
        if not iq_path.with_suffix(".json").is_file():
            logger.warning("Skipped (no .json next to it): %s", iq_path)
            continue
        converted.append(convert_file(iq_path, output_dir, args))
    logger.info("%d recording(s) converted into %s", len(converted), output_dir)
    return converted


if __name__ == "__main__":
    main()
