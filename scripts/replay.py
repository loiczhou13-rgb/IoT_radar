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
from iot_radar.config import DEFAULT_RADAR_CONFIG, RECORDING_DATA_DIR, load_config
from iot_radar.pipeline import build_context
from iot_radar.ui.dashboard import DashboardRadar


def _resolve_npz_path(args: argparse.Namespace) -> Path:
    """Resolve the ``.npz`` path from ``--npz`` or ``--subset``/``--index``."""
    if args.npz is not None:
        return Path(args.npz).expanduser().resolve()
    if args.subset is None or args.index is None:
        raise SystemExit(
            "Fournir --npz CHEMIN.npz ou bien --subset {train,test,val} et --index N (N ≥ 1)."
        )
    if args.index < 1:
        raise SystemExit("--index doit être >= 1.")
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
            "Configuration de l'enregistrement introuvable (%s) — "
            "utilisation de la configuration par défaut %s",
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
            "Ce .npz ne contient pas « spectrogram_db » — "
            "ré-enregistrer sans l’option « --no-spectrogram »."
        )
    spec = np.asarray(data["spectrogram_db"], dtype=np.float64)
    n_spec = spec.shape[0]
    if n_spec == 0:
        raise SystemExit("Aucune trame exploitable dans le .npz.")

    if "n_trame" in data.files:
        ntr = np.asarray(data["n_trame"], dtype=np.int64).ravel()
        n = int(min(n_spec, ntr.size))
    else:
        ntr = None
        n = n_spec

    frames: list[dict[str, Any]] = []
    for i in range(n):
        n_trame_val = int(ntr[i]) if ntr is not None and ntr.size > i else i + 1
        frames.append(
            {
                "spectre_colonne": np.asarray(spec[i], dtype=np.float64),
                "n_trame": n_trame_val,
            },
        )
    return frames, n


def main() -> None:
    """CLI entry point: replay a recorded ``.npz`` in the dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Visualiser un .npz (record_acquisition) — dashboard comme main.py.",
    )
    p.add_argument("--npz", type=Path, default=None, help="Fichier .npz (prioritaire sur --subset/--index).")
    p.add_argument(
        "--subset",
        choices=("train", "test", "val"),
        default=None,
        metavar="SUBSET",
        help="Sous-dossier sous data/ (avec --index).",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help="Indice échantillon (≥1), avec --subset.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Racine data (défaut : {RECORDING_DATA_DIR}).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML (prioritaire sur config_path dans le .npz puis .json hérité).",
    )
    p.add_argument(
        "--interval-ms",
        type=float,
        default=30.0,
        help="Intervalle entre trames replay (ms).",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Reboucler la lecture.",
    )
    args = p.parse_args()

    npz_path = _resolve_npz_path(args)
    if not npz_path.is_file():
        raise SystemExit(f"Fichier introuvable : {npz_path}")

    stored_config_path, rec_label = read_recording_metadata(npz_path)
    cfg_path = _choose_config_path(args.config, stored_config_path)

    cfg: dict[str, Any] = load_config(cfg_path)
    context = build_context(cfg)

    with np.load(npz_path, allow_pickle=False) as data:
        if "f_hz" in data:
            context = {**context, "f_hz": np.asarray(data["f_hz"], dtype=np.float64)}
        frames, n_tr = _frames_from_npz(data)

    logging.info(
        "Relecture — %d trames — label=%s — %s — config %s",
        n_tr,
        rec_label if rec_label is not None else "?",
        npz_path,
        cfg_path,
    )

    title = "Radar Micro-Doppler — Relecture enregistrement"
    if rec_label is not None:
        title = f"{title}    ·    label {rec_label}"
    dashboard = DashboardRadar(
        config=cfg, context=context, show_presence_score=False, title=title,
    )
    replay_frames = itertools.cycle(frames) if args.loop else frames
    dashboard.run(replay_frames, frame_interval_s=max(1.0, float(args.interval_ms)) / 1000.0)

if __name__ == "__main__":
    main()
