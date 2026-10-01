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

DEFAULT_TRAINING_CONFIG: Path = REPO_ROOT / "AICalibration" / "config.yaml"
"""Default YAML configuration of the training."""

RECORDING_DATA_DIR: Path = REPO_ROOT / "AICalibration" / "data"
"""Default root of the labelled recordings (``<train|test|val>/<n>.npz``)."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Exits the program with a clear message (exit code 1) when the file does
    not exist, and raises ``ValueError`` when it does not contain a mapping.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        print(f"ERREUR : fichier de configuration introuvable : {path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config invalide (pas un mapping) : {path}")
    return cfg


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path written in a configuration file.

    Relative paths are relative to the repository root, so that a command
    gives the same result whatever the current working directory.
    """
    pth = Path(path).expanduser()
    if pth.is_absolute():
        return pth.resolve()
    return (REPO_ROOT / pth).resolve()


def radar_log_file(cfg: dict[str, Any], explicit: str | Path | None = None) -> Path | None:
    """Log file of a radar run, from the ``logging`` section of the config.

    Returns ``None`` when ``logging.to_file`` is false (default: true), even
    if *explicit* is given; otherwise *explicit* (the ``--log-file`` option)
    or a timestamped file ``LOGS_DIR/radar_<YYYYmmdd_HHMMSS>.log``.
    """
    log_cfg = cfg.get("logging", {})
    if not bool(log_cfg.get("to_file", True)):
        return None
    if explicit is not None:
        return Path(explicit)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"radar_{stamp}.log"


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    console_level: str | None = None,
    file_mode: str = "w",
) -> None:
    """Configure the root logger (the only logging set-up of the project).

    A console handler is always installed; when *log_file* is given, a file
    handler writes the same records to disk so that runs can be audited
    offline.  Previously installed root handlers are removed.

    Parameters
    ----------
    level : str, optional
        Level name of the root logger and of the file (``"DEBUG"``,
        ``"INFO"``, ``"WARNING"``…).
    log_file : pathlib.Path or None, optional
        File receiving the records (its folder is created if needed).
    console_level : str or None, optional
        Level of the console handler; defaults to *level*.  The training
        uses ``"WARNING"`` to keep the console for its progress bar.
    file_mode : str, optional
        ``"w"`` (overwrite, default) or ``"a"`` (append).
    """
    level_value = getattr(logging, str(level).upper(), logging.INFO)
    console_value = (
        level_value if console_level is None
        else getattr(logging, str(console_level).upper(), logging.INFO)
    )
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    root = logging.getLogger()
    root.setLevel(level_value)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(console_value)
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_file is None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(level_value)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)
    logger.info("Journal écrit dans %s", log_file)
