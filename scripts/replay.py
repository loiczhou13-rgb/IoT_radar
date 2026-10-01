#!/usr/bin/env python3
"""Offline replay of a recording — same RX/TX dashboard as the live radar,
without the "presence score" panel (the .npz does not carry those series).

Examples (from the repository root) ::

    python scripts/replay.py --subset train --index 1

    python scripts/replay.py --npz AICalibration/data/train/3.npz

The ``.npz`` must contain ``spectrogram_db``. YAML path: ``--config``, the
``config_path`` key inside the ``.npz``, or otherwise the ``.json`` of the same index.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np

from iot_radar.acquisition.recording import read_recording_metadata
from iot_radar.config import DEFAULT_RADAR_CONFIG, RECORDING_DATA_DIR, load_config, setup_logging
from iot_radar.pipeline import build_context
from iot_radar.ui.dashboard import DashboardRadar


def _resolve_npz_path(args: argparse.Namespace) -> Path:
    """Resolve the ``.npz`` path from ``--npz`` or ``--subset``/``--index``."""
    if args.npz is not None:
        return Path(args.npz).expanduser().resolve()
    if args.subset is None or args.index is None:
        raise SystemExit(
            "Give --npz PATH.npz, or --subset {train,test,val} and --index N (N >= 1)."
        )
    if args.index < 1:
        raise SystemExit("--index must be >= 1.")
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else RECORDING_DATA_DIR
    )
    return (data_root / args.subset / f"{int(args.index)}.npz").resolve()


def _choose_config_path(cli_config: str | None, stored_config_path: str | None) -> str:
    """YAML used for the replay: ``--config``, else the recorded one, else the default.

    Recordings store the **absolute** path of their configuration file.  When
    that file no longer exists (repository moved or reorganised), the default
    configuration is used instead, with a warning.
    """
    if cli_config is not None:
        return str(Path(cli_config).expanduser().resolve())
    if stored_config_path and Path(stored_config_path).expanduser().is_file():
        return str(Path(stored_config_path).expanduser().resolve())
    if stored_config_path:
        logging.warning(
            "Configuration of the recording not found (%s) — "
            "using the default configuration %s",
            stored_config_path,
            DEFAULT_RADAR_CONFIG,
        )
    return str(DEFAULT_RADAR_CONFIG)


def _frames_from_npz(data: Any) -> tuple[list[dict[str, Any]], int]:
    """Build replay frames from a recording's ``spectrogram_db`` array.

    Returns ``(frames, n_frames)``.
    """
    if "spectrogram_db" not in data:
        raise SystemExit(
            "This .npz has no 'spectrogram_db' — record it again "
            "without the --no-spectrogram option."
        )
    spectrogram_db = np.asarray(data["spectrogram_db"], dtype=np.float64)
    n_columns = spectrogram_db.shape[0]
    if n_columns == 0:
        raise SystemExit("No usable frame in the .npz.")

    # "n_trame" is the key of the frame numbers in the legacy .npz format.
    if "n_trame" in data.files:
        frame_numbers = np.asarray(data["n_trame"], dtype=np.int64).ravel()
        n = int(min(n_columns, frame_numbers.size))
    else:
        frame_numbers = None
        n = n_columns

    frames: list[dict[str, Any]] = []
    for i in range(n):
        if frame_numbers is not None and frame_numbers.size > i:
            frame_number = int(frame_numbers[i])
        else:
            frame_number = i + 1
        frames.append(
            {
                "spectrum_column_db": np.asarray(spectrogram_db[i], dtype=np.float64),
                "frame_number": frame_number,
            },
        )
    return frames, n


def main() -> None:
    """CLI entry point: replay a recorded ``.npz`` in the dashboard."""
    setup_logging("INFO")

    p = argparse.ArgumentParser(
        description="Replay a recorded .npz in the radar dashboard.",
    )
    p.add_argument("--npz", type=Path, default=None, help=".npz file (takes precedence over --subset/--index).")
    p.add_argument(
        "--subset",
        choices=("train", "test", "val"),
        default=None,
        metavar="SUBSET",
        help="Sub-folder of the data root (with --index).",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help="Sample index (>= 1), with --subset.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Data root (default: {RECORDING_DATA_DIR}).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML (takes precedence over the config path stored in the .npz or .json).",
    )
    p.add_argument(
        "--interval-ms",
        type=float,
        default=30.0,
        help="Interval between replayed frames (ms).",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Replay in a loop.",
    )
    args = p.parse_args()

    npz_path = _resolve_npz_path(args)
    if not npz_path.is_file():
        raise SystemExit(f"File not found: {npz_path}")

    stored_config_path, rec_label = read_recording_metadata(npz_path)
    cfg_path = _choose_config_path(args.config, stored_config_path)

    cfg: dict[str, Any] = load_config(cfg_path)
    context = build_context(cfg)

    with np.load(npz_path, allow_pickle=False) as data:
        if "f_hz" in data:
            context = {**context, "f_hz": np.asarray(data["f_hz"], dtype=np.float64)}
        frames, n_frames = _frames_from_npz(data)

    logging.info(
        "Replay — %d frames — label=%s — %s — config %s",
        n_frames,
        rec_label if rec_label is not None else "?",
        npz_path,
        cfg_path,
    )

    title = "Micro-Doppler radar — recording replay"
    if rec_label is not None:
        title = f"{title}    ·    label {rec_label}"
    dashboard = DashboardRadar(
        config=cfg, context=context, show_presence_score=False, title=title,
    )
    replay_frames = itertools.cycle(frames) if args.loop else frames
    dashboard.run(replay_frames, frame_interval_s=max(1.0, float(args.interval_ms)) / 1000.0)

if __name__ == "__main__":
    main()
