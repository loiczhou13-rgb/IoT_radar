"""Recording (simulation) and replay helpers of the legacy .npz format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from characterization_cases import pipeline_config
from iot_radar.acquisition.recording import next_sample_index, read_recording_metadata
from script_loader import load_script


def test_record_and_replay_helpers(tmp_path: Path) -> None:
    config_path = tmp_path / "radar.yaml"
    config_path.write_text(yaml.safe_dump(pipeline_config()), encoding="utf-8")
    data_root = tmp_path / "data"
    record = load_script("record")
    replay = load_script("replay")
    args = record.build_argument_parser().parse_args([
        "--subset", "train", "--env", "salle", "--label", "1", "--duration", "1.0",
        "--config", str(config_path), "--simulation", "--data-root", str(data_root),
    ])
    npz_path = record.run(args)

    assert npz_path == (data_root / "train" / "1.npz").resolve()
    with np.load(npz_path) as z:
        n_frames = int(z["n_frames"])
        assert n_frames > 0
        assert z["spectrogram_db"].shape == (n_frames, 4096)
        assert int(z["label"]) == 1 and str(z["env"]) == "salle"
        assert float(z["f_s_dec_hz"]) == 1000.0
        n_iq = int(z["iq_num_complex_samples"])
    meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["label"] == 1 and meta["n_frames"] == n_frames
    iq = np.fromfile(npz_path.with_suffix(".iq"), dtype=np.complex64)
    assert iq.size == n_iq > 0

    # Next free index.
    assert next_sample_index(npz_path.parent) == 2

    # Replay helpers.
    assert read_recording_metadata(npz_path) == (str(config_path.resolve()), 1)
    with np.load(npz_path) as z:
        frames, n = replay._frames_from_npz(z)
    assert n == n_frames and len(frames) == n_frames
    assert frames[0]["spectrum_column_db"].shape == (4096,)


def test_metadata_falls_back_on_json_sidecar(tmp_path: Path) -> None:
    npz_path = tmp_path / "1.npz"
    np.savez_compressed(npz_path, spectrogram_db=np.zeros((2, 4)))
    assert read_recording_metadata(npz_path) == (None, None)
    npz_path.with_suffix(".json").write_text(json.dumps({"config_path": "/x.yaml", "label": 0}), encoding="utf-8")
    assert read_recording_metadata(npz_path) == ("/x.yaml", 0)


def test_record_refuses_index_with_several_samples(tmp_path: Path) -> None:
    import pytest

    record = load_script("record")
    with pytest.raises(SystemExit):
        record.main(["--subset", "train", "--env", "salle", "--label", "1", "-n", "2", "--index", "3"])


def test_replay_config_falls_back_to_default_when_recorded_path_is_gone(tmp_path: Path) -> None:
    from iot_radar.config import DEFAULT_RADAR_CONFIG

    replay = load_script("replay")
    existing = tmp_path / "radar.yaml"
    existing.write_text("{}", encoding="utf-8")
    assert replay._choose_config_path(None, str(existing)) == str(existing.resolve())
    assert replay._choose_config_path(None, "/no/such/config.yaml") == str(DEFAULT_RADAR_CONFIG)
    assert replay._choose_config_path(None, None) == str(DEFAULT_RADAR_CONFIG)
    assert replay._choose_config_path(str(existing), "/no/such/config.yaml") == str(existing.resolve())
