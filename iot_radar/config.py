"""Repository paths, YAML configuration loading and logging set-up.

Paths are resolved from the location of this file, so they do not depend on
the current working directory (the package is installed in editable mode with
``pip install -e .``).
"""

from __future__ import annotations

import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
"""Root of the IoT_radar repository."""

DEFAULT_RADAR_CONFIG: Path = REPO_ROOT / "MicroDopplerDetection" / "configs" / "config.yaml"
"""Default YAML configuration of the radar."""

LOGS_DIR: Path = REPO_ROOT / "MicroDopplerDetection" / "logs"
"""Folder of the timestamped run logs."""

RECORDING_DATA_DIR: Path = REPO_ROOT / "AICalibration" / "data"
"""Default root of the labelled recordings (``<train|test|val>/<n>.npz``)."""


def load_config(path: str) -> dict[str, Any]:
    """Load and return the YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        print(f"ERREUR : fichier de configuration introuvable : {path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(cfg: dict[str, Any], log_file: Path | None = None) -> Path | None:
    """Configure the root logger from the config ``logging`` section.

    A console handler is always installed.  If *log_file* is provided
    (or if ``logging.to_file`` is true in the config), a parallel
    ``FileHandler`` writes the same records to disk so that runs can be
    audited offline.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary; reads ``logging.level`` and
        optionally ``logging.to_file`` (default: ``True``).
    log_file : pathlib.Path or None, optional
        Explicit log-file path.  When ``None``, a timestamped file is
        created under ``MicroDopplerDetection/logs/``.

    Returns
    -------
    pathlib.Path or None
        Path of the file handler (``None`` if file logging is disabled).
    """
    log_cfg = cfg.get("logging", {})
    level_name = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    write_to_file = bool(log_cfg.get("to_file", True))
    if not write_to_file:
        return None

    if log_file is None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"radar_{stamp}.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    root.addHandler(file_handler)
    logger.info("Journal écrit dans %s", log_file)
    return log_file
