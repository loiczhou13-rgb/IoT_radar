"""Repository paths, YAML configuration loading and logging set-up.

Used by every command-line script: they all load their YAML file with
:func:`load_config` and configure the logs with :func:`setup_logging`.

Paths are resolved from the location of this file, so they do not depend on
the current working directory (the package is installed in editable mode with
``pip install -e .``).
"""

from __future__ import annotations

import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
"""Root of the IoT_radar repository."""

DEFAULT_RADAR_CONFIG: Path = REPO_ROOT / "configs" / "radar.yaml"
"""Default YAML configuration of the radar."""

LOGS_DIR: Path = REPO_ROOT / "logs"
"""Folder of the timestamped run logs."""



def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Exits the program with a clear message (exit code 1) when the file does
    not exist, and raises ``ValueError`` when it does not contain a mapping.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        print(f"ERROR: configuration file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid configuration (not a mapping): {path}")
    return cfg


def check_keys(section_name: str, section: Mapping[str, Any] | None, valid_keys: Iterable[str]) -> None:
    """Refuse the keys of *section* that are not in *valid_keys*.

    Raises
    ------
    ValueError
        Naming the unknown keys, the section and the valid keys (a typo in a
        configuration file would otherwise be silently replaced by a default).
    """
    valid = sorted(valid_keys)
    unknown = sorted(set(section or {}) - set(valid))
    if unknown:
        raise ValueError(f"Unknown configuration key(s) {', '.join(unknown)} in '{section_name}'. "
                         f"Valid keys: {', '.join(valid)}.")


_SCENARIO_KEYS = {"tx_power_dbm", "tx_antenna_gain_dbi", "rx_antenna_gain_dbi", "radar_cross_section_m2",
                  "noise_figure_db", "system_losses_db", "min_snr_db"}

RADAR_CONFIG_KEYS: dict[str, Any] = {
    "logging": {"level", "to_file"},
    "sdr": {"uri", "center_frequency_hz", "sample_rate_hz", "rx_gain_db", "tx_gain_db", "buffer_size",
            "rx_kernel_buffers"},
    "tx": {"waveform", "offset_hz"},
    "slow_time": {"rate_hz", "warmup_s"},
    "vital_signs": "settings",
    "motion_robustness": "settings",
    "measurement": "settings",
    "micro_doppler_view": {"window_s", "n_fft", "window", "display_max_hz"},
    "display": {"history_s", "waterfall_columns", "full_screen"},
    "link_budget": {"optimistic": _SCENARIO_KEYS, "pessimistic": _SCENARIO_KEYS, "noise_bandwidth_hz": None},
    "recording": {"sessions_dir", "flush_interval_s"},
    "simulation": {"enabled", "presence", "timeline", "breath_rate_hz", "breath_amplitude_mm", "heart_rate_hz",
                   "heart_amplitude_mm", "motion_amplitude_mm", "target_range_m", "target_amplitude",
                   "static_clutter_amplitude", "receiver_dc_amplitude", "lo_offset_hz", "snr_db", "seed",
                   "realtime", "static_scatterers", "disturbers"},
}
"""Valid keys of ``configs/radar.yaml``: a set of leaf keys, a mapping for nested
sections, ``None`` for a leaf, ``"settings"`` for the sections checked by
their settings class (``VitalSignsSettings``, ``MotionRobustnessSettings``,
``MeasurementSettings``)."""


def check_radar_config(cfg: Mapping[str, Any]) -> None:
    """Refuse any unknown key of a radar configuration (see :data:`RADAR_CONFIG_KEYS`)."""

    def check(name: str, section: Any, valid: Any) -> None:
        if valid is None or valid == "settings" or not isinstance(section, Mapping):
            return
        check_keys(name or "radar configuration", section, valid)
        if isinstance(valid, Mapping):
            for key, value in section.items():
                check(f"{name}.{key}" if name else key, value, valid[key])

    check("", cfg, RADAR_CONFIG_KEYS)


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


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger (the only logging set-up of the project).

    A console handler is always installed; when *log_file* is given, a file
    handler writes the same records to disk so that runs can be audited
    offline.  Previously installed root handlers are removed.

    Parameters
    ----------
    level : str, optional
        Level name of the root logger, the console and the file
        (``"DEBUG"``, ``"INFO"``, ``"WARNING"``…).
    log_file : pathlib.Path or None, optional
        File receiving the records (its folder is created if needed).
    """
    level_value = getattr(logging, str(level).upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    root = logging.getLogger()
    root.setLevel(level_value)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(level_value)
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_file is None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level_value)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)
    logger.info("Log file: %s", log_file)
