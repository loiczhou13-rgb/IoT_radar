"""Replay helpers of the legacy .npz format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from iot_radar.acquisition.recording import read_recording_metadata
from script_loader import load_script


def test_metadata_falls_back_on_json_sidecar(tmp_path: Path) -> None:
    npz_path = tmp_path / "1.npz"
    np.savez_compressed(npz_path, spectrogram_db=np.zeros((2, 4)))
    assert read_recording_metadata(npz_path) == (None, None)
    npz_path.with_suffix(".json").write_text(json.dumps({"config_path": "/x.yaml", "label": 0}), encoding="utf-8")
    assert read_recording_metadata(npz_path) == ("/x.yaml", 0)


def test_replay_config_falls_back_to_default_when_recorded_path_is_gone(tmp_path: Path) -> None:
    from iot_radar.config import DEFAULT_RADAR_CONFIG

    replay = load_script("replay")
    existing = tmp_path / "radar.yaml"
    existing.write_text("{}", encoding="utf-8")
    assert replay._choose_config_path(None, str(existing)) == str(existing.resolve())
    assert replay._choose_config_path(None, "/no/such/config.yaml") == str(DEFAULT_RADAR_CONFIG)
    assert replay._choose_config_path(None, None) == str(DEFAULT_RADAR_CONFIG)
    assert replay._choose_config_path(str(existing), "/no/such/config.yaml") == str(existing.resolve())
