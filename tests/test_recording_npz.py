"""Recording (simulation) and replay helpers of the legacy .npz format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from characterization_cases import pipeline_config
from MicroDopplerDetection.utils import record_acquisition, record_visualization


def test_record_and_replay_helpers(tmp_path: Path) -> None:
    config_path = tmp_path / "radar.yaml"
    config_path.write_text(yaml.safe_dump(pipeline_config()), encoding="utf-8")
    data_root = tmp_path / "data"
    args = record_acquisition.parse_record_args([
        "--subset", "train", "--env", "salle", "--label", "1", "--duration", "1.0",
        "--config", str(config_path), "--simulation", "--data-root", str(data_root),
    ])
    npz_path = record_acquisition.run(args)

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
    assert record_acquisition._next_sample_index(npz_path.parent) == 2

    # Replay helpers.
    assert record_visualization._load_config_path_from_recording(npz_path) == str(config_path.resolve())
    assert record_visualization._label_from_json_sidecar(npz_path) == 1
    with np.load(npz_path) as z:
        frames, n = record_visualization._frames_from_npz(z)
    assert n == n_frames and len(frames) == n_frames
    assert frames[0]["spectre_colonne"].shape == (4096,)
