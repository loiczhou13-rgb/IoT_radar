"""Radar sessions recorded in HDF5 files (schema v1), and their reader.

Place in the chain: next to the acquisition.  A session stores the **raw IQ
blocks** delivered by a source (PlutoSDR or simulation), so that a replay
(:class:`iot_radar.acquisition.sources.ReplaySource`) feeds the pipeline with
exactly the same samples as the live run.

File layout (one file per session, never modified after the acquisition)::

    /                 attributes: schema_version, sample_rate_hz,
                      center_frequency_hz, tx_waveform, tx_offset_hz,
                      tx_gain_db, rx_gain_db, rx_gain_mode, n_rx_channels,
                      channel_layout, firmware_version, source_kind, iq_stage,
                      start_time_utc, software_version, config_yaml and the
                      scene (subject_id, distance_m, orientation, obstacle,
                      obstacle_thickness_cm, room, notes)
    /iq               int16, (n_channels, n_samples, 2): I and Q, ADC units x scale
    /blocks           table (sample_start, host_time_s, overflow), one row per block
    /annotations      table (sample_start, sample_count, label)
    /ground_truth/    optional: time_s and chest_displacement_m

Robustness: the file is created in SWMR mode (single writer, multiple
readers) and flushed every ``flush_interval_s`` seconds, so it stays readable
up to the last flush if the program is killed.  It is opened with mode
``"w-"`` (an existing file is never overwritten) and made read-only when
closed.

Missing metadata are stored as ``""`` (text) or ``NaN`` (numbers).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import iot_radar
from iot_radar.acquisition.pluto import effective_tx_offset_hz

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
"""Version of the session file layout described in this module."""

LABELS: tuple[str, ...] = ("empty", "breathing", "apnea", "motion", "unknown")
"""Annotation vocabulary.

* ``empty`` — nobody in the field of the radar;
* ``breathing`` — one person present, still, breathing;
* ``apnea`` — one person present, still, without breathing movement;
* ``motion`` — one person present whose body movements dominate breathing;
* ``unknown`` — not annotated or uncertain segment.
"""

RADAR_ATTRIBUTES: tuple[str, ...] = (
    "sample_rate_hz",
    "center_frequency_hz",
    "tx_waveform",
    "tx_offset_hz",
    "tx_gain_db",
    "rx_gain_db",
    "rx_gain_mode",
    "n_rx_channels",
    "channel_layout",
    "firmware_version",
    "source_kind",
    "iq_stage",
)
"""Attributes describing the radar, required by :class:`SessionWriter`."""

SCENE_DEFAULTS: dict[str, Any] = {
    "subject_id": "",
    "distance_m": math.nan,
    "orientation": "",
    "obstacle": "",
    "obstacle_thickness_cm": math.nan,
    "room": "",
    "notes": "",
}
"""Scene attributes and the values stored when they are unknown."""

IQ_CHUNK_SAMPLES: int = 65536
"""Length of the HDF5 chunks of ``/iq`` along time (256 kB per channel)."""

BLOCKS_DTYPE = np.dtype([("sample_start", "<i8"), ("host_time_s", "<f8"), ("overflow", "u1")])
ANNOTATIONS_DTYPE = np.dtype([("sample_start", "<i8"), ("sample_count", "<i8"), ("label", "S16")])

_SESSION_NAME = re.compile(r"^session_(\d+)_\d{8}_\d{6}\.h5$")


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------

def next_session_id(sessions_dir: Path) -> int:
    """Largest session identifier found in *sessions_dir*, plus one (1 if none)."""
    if not Path(sessions_dir).is_dir():
        return 1
    identifiers = [
        int(match.group(1))
        for match in (_SESSION_NAME.match(p.name) for p in Path(sessions_dir).iterdir())
        if match is not None
    ]
    return max(identifiers, default=0) + 1


def session_file_name(session_id: int, start_time_utc: dt.datetime) -> str:
    """``session_<id on 4 digits>_<YYYYMMDD>_<HHMMSS>.h5`` (start time in UTC)."""
    return f"session_{session_id:04d}_{start_time_utc.strftime('%Y%m%d_%H%M%S')}.h5"


def radar_attributes(cfg: dict[str, Any], source: Any) -> dict[str, Any]:
    """Radar attributes of a session (:data:`RADAR_ATTRIBUTES`).

    Parameters
    ----------
    cfg : dict
        Configuration used for the acquisition (``sdr`` and ``tx`` sections).
    source : Source
        The IQ source being recorded (rate, channels, kind, firmware).
    """
    sdr = cfg["sdr"]
    return {
        "sample_rate_hz": float(source.sample_rate_hz),
        "center_frequency_hz": float(sdr["center_frequency_hz"]),
        "tx_waveform": str(cfg["tx"]["waveform"]),
        "tx_offset_hz": effective_tx_offset_hz(cfg),
        "tx_gain_db": float(sdr["tx_gain_db"]),
        "rx_gain_db": float(sdr["rx_gain_db"]),
        "rx_gain_mode": "manual",
        "n_rx_channels": int(source.n_channels),
        "channel_layout": ",".join(f"rx{i}" for i in range(source.n_channels)),
        "firmware_version": str(source.firmware_version),
        "source_kind": str(source.kind),
        "iq_stage": "raw_adc",
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class SessionWriter:
    """Write one radar session, block by block.

    Parameters
    ----------
    path : pathlib.Path
        File to create (it must not exist).
    radar : dict
        Radar attributes, with every key of :data:`RADAR_ATTRIBUTES`.
    scene : dict, optional
        Scene attributes (keys of :data:`SCENE_DEFAULTS`); missing ones are
        stored as unknown.
    config_yaml : str, optional
        Text of the YAML configuration used during the acquisition.
    start_time_utc : datetime, optional
        Start of the session (default: now).
    iq_scale : float, optional
        ADC units per stored unit: ``1.0`` for raw ADC samples, which are
        integers.  Other data are rounded to the nearest multiple of
        *iq_scale*.
    with_ground_truth : bool, optional
        Create the ``/ground_truth`` group (filled by :meth:`set_ground_truth`).
    flush_interval_s : float, optional
        Maximum time between two flushes to disk (s).
    extra_attributes : dict, optional
        Other root attributes (e.g. ``legacy_source_file``).

    Notes
    -----
    In SWMR mode no object or attribute can be created once writing has
    started: all the datasets are created here, the annotations and the
    ground truth are kept in memory and written by :meth:`close`.
    """

    def __init__(
        self,
        path: Path,
        radar: dict[str, Any],
        scene: dict[str, Any] | None = None,
        config_yaml: str = "",
        start_time_utc: dt.datetime | None = None,
        iq_scale: float = 1.0,
        with_ground_truth: bool = False,
        flush_interval_s: float = 1.0,
        extra_attributes: dict[str, Any] | None = None,
    ) -> None:
        missing = [key for key in RADAR_ATTRIBUTES if key not in radar]
        if missing:
            raise ValueError(f"Missing radar attributes: {missing}")
        unknown_scene = sorted(set(scene or {}) - set(SCENE_DEFAULTS))
        if unknown_scene:
            raise ValueError(f"Unknown scene attributes: {unknown_scene}")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_channels = int(radar["n_rx_channels"])
        self._iq_scale = float(iq_scale)
        self._flush_interval_s = float(flush_interval_s)
        self._annotations: list[tuple[int, int, str]] = []
        self._ground_truth: tuple[np.ndarray, np.ndarray] | None = None
        self._first_host_time_s: float | None = None
        self._last_flush_s = time.monotonic()
        self.n_samples = 0
        self.n_blocks = 0

        start = start_time_utc or dt.datetime.now(dt.timezone.utc)
        attributes: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "start_time_utc": start.isoformat(),
            "software_version": iot_radar.__version__,
            "config_yaml": config_yaml,
            **radar,
            **SCENE_DEFAULTS,
            **(scene or {}),
            **(extra_attributes or {}),
        }

        self._file = h5py.File(self.path, "w-", libver="latest")
        for key, value in attributes.items():
            self._file.attrs[key] = value

        iq = self._file.create_dataset(
            "iq",
            shape=(self.n_channels, 0, 2),
            maxshape=(self.n_channels, None, 2),
            chunks=(self.n_channels, IQ_CHUNK_SAMPLES, 2),
            dtype=np.int16,
        )
        iq.attrs["scale"] = self._iq_scale
        iq.attrs["unit"] = "adc_lsb"
        iq.attrs["layout"] = "(channel, sample, [I, Q])"
        self._file.create_dataset("blocks", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=BLOCKS_DTYPE)
        self._file.create_dataset("annotations", shape=(0,), maxshape=(None,), chunks=(64,), dtype=ANNOTATIONS_DTYPE)
        if with_ground_truth:
            group = self._file.create_group("ground_truth")
            for name in ("time_s", "chest_displacement_m"):
                group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(4096,), dtype=np.float64)
            group.attrs["source"] = "simulation"
            group.attrs["description"] = "simulated radial chest displacement (positive away from the radar)"
        self._file.swmr_mode = True
        logger.info("Session file created: %s", self.path)

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write_block(self, samples: np.ndarray, host_time_s: float, overflow: bool) -> None:
        """Append one block of samples.

        Parameters
        ----------
        samples : numpy.ndarray
            Complex samples, shape ``(n_channels, n_samples)``, in ADC units.
        host_time_s : float
            Host clock when the block was received (s); stored relative to
            the first block.
        overflow : bool
            ``True`` if samples were lost just before this block.
        """
        samples = np.asarray(samples)
        if samples.ndim != 2 or samples.shape[0] != self.n_channels:
            raise ValueError(f"Expected samples of shape ({self.n_channels}, n), got {samples.shape}.")
        n = samples.shape[1]
        stored = np.empty((self.n_channels, n, 2), dtype=np.int16)
        stored[:, :, 0] = _to_int16(samples.real / self._iq_scale)
        stored[:, :, 1] = _to_int16(samples.imag / self._iq_scale)

        iq = self._file["iq"]
        iq.resize(self.n_samples + n, axis=1)
        iq[:, self.n_samples:self.n_samples + n, :] = stored

        if self._first_host_time_s is None:
            self._first_host_time_s = float(host_time_s)
        blocks = self._file["blocks"]
        blocks.resize(self.n_blocks + 1, axis=0)
        blocks[self.n_blocks] = (self.n_samples, float(host_time_s) - self._first_host_time_s, int(bool(overflow)))

        self.n_samples += n
        self.n_blocks += 1
        if time.monotonic() - self._last_flush_s >= self._flush_interval_s:
            self.flush()

    def add_annotation(self, sample_start: int, sample_count: int, label: str) -> None:
        """Label the samples ``[sample_start, sample_start + sample_count)`` (written on close)."""
        if label not in LABELS:
            raise ValueError(f"Unknown label '{label}', use one of {LABELS}.")
        self._annotations.append((int(sample_start), int(sample_count), label))

    def set_ground_truth(self, time_s: np.ndarray, chest_displacement_m: np.ndarray) -> None:
        """Reference displacement of the chest and its time axis (written on close)."""
        if "ground_truth" not in self._file:
            raise RuntimeError("The session was created without ground truth (with_ground_truth=False).")
        self._ground_truth = (np.asarray(time_s, np.float64), np.asarray(chest_displacement_m, np.float64))

    def flush(self) -> None:
        """Write the buffered data to disk."""
        self._file.flush()
        self._last_flush_s = time.monotonic()

    def close(self) -> None:
        """Write the annotations and the ground truth, close and protect the file."""
        if not self._file:
            return
        annotations = self._file["annotations"]
        annotations.resize(len(self._annotations), axis=0)
        for i, (start, count, label) in enumerate(self._annotations):
            annotations[i] = (start, count, label.encode("ascii"))
        if self._ground_truth is not None:
            time_s, displacement_m = self._ground_truth
            group = self._file["ground_truth"]
            for name, values in (("time_s", time_s), ("chest_displacement_m", displacement_m)):
                group[name].resize(values.size, axis=0)
                group[name][:] = values
        self._file.flush()
        self._file.close()
        os.chmod(self.path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only
        logger.info("Session closed: %s (%d samples, %d blocks)", self.path, self.n_samples, self.n_blocks)


def _to_int16(values: np.ndarray) -> np.ndarray:
    """Round to the nearest integer, refusing values outside the int16 range."""
    rounded = np.rint(values)
    if rounded.size and (rounded.min() < -32768 or rounded.max() > 32767):
        raise ValueError("Sample outside the int16 range: increase iq_scale.")
    return rounded.astype(np.int16)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class SessionReader:
    """Read a session file written by :class:`SessionWriter`.

    A session whose writer was killed can be read too: it holds the samples
    flushed before the interruption, but no annotation nor ground truth.

    Parameters
    ----------
    path : pathlib.Path
        Session file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # swmr=True also opens a session whose writer was killed (the file
        # then still carries the "open for writing" flag).
        self._file = h5py.File(self.path, "r", swmr=True)
        version = int(self._file.attrs.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            self._file.close()
            raise ValueError(f"{self.path}: unsupported schema_version {version} (expected {SCHEMA_VERSION}).")
        self.attributes: dict[str, Any] = {key: _python_value(v) for key, v in self._file.attrs.items()}
        self._iq = self._file["iq"]
        self.n_channels, self.n_samples = int(self._iq.shape[0]), int(self._iq.shape[1])
        self.iq_scale = float(self._iq.attrs["scale"])
        self.sample_rate_hz = float(self.attributes["sample_rate_hz"])

    def __enter__(self) -> "SessionReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_iq(self, start: int, stop: int) -> np.ndarray:
        """Samples ``[start, stop)`` as complex64, shape ``(n_channels, stop - start)``, ADC units."""
        stored = self._iq[:, start:stop, :].astype(np.float32)
        return ((stored[:, :, 0] + 1j * stored[:, :, 1]) * self.iq_scale).astype(np.complex64)

    def blocks(self) -> np.ndarray:
        """Table of the received blocks (structured array, see :data:`BLOCKS_DTYPE`)."""
        return self._file["blocks"][:]

    def annotations(self) -> list[tuple[int, int, str]]:
        """``(sample_start, sample_count, label)`` of every annotated segment."""
        return [(int(r["sample_start"]), int(r["sample_count"]), r["label"].decode("ascii"))
                for r in self._file["annotations"][:]]

    def ground_truth(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(time_s, chest_displacement_m)``, or ``None`` if the session has none."""
        if "ground_truth" not in self._file:
            return None
        group = self._file["ground_truth"]
        return group["time_s"][:], group["chest_displacement_m"][:]

    def close(self) -> None:
        """Close the file."""
        self._file.close()


def _python_value(value: Any) -> Any:
    """HDF5 attribute → plain Python value (str, int, float)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Legacy .npz recordings (read by the replay until it switches to sessions)
# ---------------------------------------------------------------------------

def read_recording_metadata(npz_path: Path) -> tuple[str | None, int | None]:
    """Return ``(config_path, label)`` of a recording.

    Each value is read from the ``.npz`` first and, when missing there, from
    the ``.json`` sidecar written next to it.  A missing or unreadable value
    is returned as ``None``.
    """
    config_path: str | None = None
    label: int | None = None
    with np.load(npz_path, allow_pickle=False) as data:
        if "config_path" in data.files:
            config_path = str(np.asarray(data["config_path"]).item())
        if "label" in data.files:
            label = int(np.asarray(data["label"]).item())
    if config_path is not None and label is not None:
        return config_path, label

    sidecar = npz_path.with_suffix(".json")
    if not sidecar.is_file():
        return config_path, label
    try:
        with open(sidecar, encoding="utf-8") as fh:
            meta: dict[str, Any] = json.load(fh)
    except json.JSONDecodeError:
        logger.warning("Unreadable JSON — %s", sidecar)
        return config_path, label
    if config_path is None:
        stored = meta.get("config_path")
        config_path = str(stored) if isinstance(stored, str) and stored else None
    if label is None:
        try:
            stored_label = meta.get("label")
            label = int(stored_label) if stored_label is not None else None
        except (TypeError, ValueError):
            label = None
    return config_path, label
