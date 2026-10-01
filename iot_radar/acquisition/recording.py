"""Files written by an acquisition and read back by the replay (legacy .npz format).

Place in the chain: next to the acquisition — a recording stores what the
pipeline computed during a session.

A recording ``<n>`` of the subset ``<train|test|val>`` is made of
``<data-root>/<subset>/<n>.npz`` (spectrogram + metadata), ``<n>.json``
(the same metadata, human-readable) and optionally ``<n>.iq`` (raw complex64)
or ``<n>.wav`` (stereo float32 I/Q).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def write_iq_complex64(path: Path, iq: np.ndarray) -> None:
    """Write IQ samples to a raw ``.iq`` file (dtype complex64 / float32×2)."""
    if iq.size == 0:
        raise ValueError("Empty IQ signal — cannot write the .iq file.")
    z = np.asarray(iq, dtype=np.complex64)
    z.tofile(path)


def write_iq_stereo_wav(path: Path, iq: np.ndarray, sample_rate_hz: float) -> None:
    """Write complex IQ to a stereo float32 WAV (I = channel 0, Q = channel 1)."""
    if iq.size == 0:
        raise ValueError("Empty IQ signal — cannot write the WAV file.")
    i = np.asarray(iq.real, dtype=np.float32)
    q = np.asarray(iq.imag, dtype=np.float32)
    stereo = np.column_stack((i, q))
    wavfile.write(path, int(round(float(sample_rate_hz))), stereo)


def next_sample_index(split_dir: Path) -> int:
    """Largest existing index ``n`` (``n.npz`` files) + 1, or 1 if empty."""
    if not split_dir.is_dir():
        return 1
    best = 0
    for p in split_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".npz":
            continue
        try:
            n = int(p.stem)
        except ValueError:
            continue
        if n >= 1:
            best = max(best, n)
    return best + 1


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
