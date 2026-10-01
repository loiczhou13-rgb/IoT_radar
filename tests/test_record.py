"""scripts/record.py: simulated sessions written to HDF5 files."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from characterization_cases import pipeline_config
from iot_radar.acquisition.recording import SessionReader
from script_loader import load_script


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    cfg = pipeline_config()
    cfg["recording"]["sessions_dir"] = str(tmp_path / "sessions")
    path = tmp_path / "radar.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_simulated_session(config_path: Path, tmp_path: Path) -> None:
    record = load_script("record")
    (path,) = record.main([
        "--simulation", "--duration-s", "0.2", "--config", str(config_path),
        "--room", "lab", "--distance-m", "2.5", "--subject-id", "S01",
    ])
    assert path.parent == tmp_path / "sessions"
    assert path.name.startswith("session_0001_") and path.suffix == ".h5"
    with SessionReader(path) as session:
        attrs = session.attributes
        assert attrs["source_kind"] == "simulation" and attrs["iq_stage"] == "raw_adc"
        assert attrs["tx_offset_hz"] == 244.140625 and attrs["sample_rate_hz"] == 100e3
        assert attrs["room"] == "lab" and attrs["distance_m"] == 2.5 and attrs["subject_id"] == "S01"
        assert math.isnan(attrs["obstacle_thickness_cm"])
        assert yaml.safe_load(attrs["config_yaml"])["tx"]["waveform"] == "cw_offset"
        assert session.n_samples >= 20_000 and session.n_samples % 4096 == 0  # whole blocks
        assert session.annotations() == [(0, session.n_samples, "breathing")]
        time_s, displacement_m = session.ground_truth()
        assert time_s[1] == pytest.approx(0.01) and abs(displacement_m).max() <= 0.010


def test_label_sets_the_simulated_scene_and_ids_follow(config_path: Path) -> None:
    record = load_script("record")
    paths = record.main(["--simulation", "--label", "empty", "--duration-s", "0.05",
                         "--config", str(config_path), "-n", "2"])
    assert [p.name[:12] for p in paths] == ["session_0001", "session_0002"]
    with SessionReader(paths[1]) as session:
        assert session.annotations()[0][2] == "empty"
        assert abs(session.ground_truth()[1]).max() == 0.0


def test_command_line_checks(config_path: Path, tmp_path: Path) -> None:
    record = load_script("record")
    hardware_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hardware_cfg["simulation"]["enabled"] = False
    hardware_path = tmp_path / "hardware.yaml"
    hardware_path.write_text(yaml.safe_dump(hardware_cfg), encoding="utf-8")
    with pytest.raises(SystemExit):  # the PlutoSDR needs a label
        record.main(["--config", str(hardware_path), "--duration-s", "1"])
    with pytest.raises(SystemExit):
        record.main(["--simulation", "--config", str(config_path), "-n", "2", "--session-id", "4"])
    with pytest.raises(SystemExit):  # not reproducible by the simulation
        record.main(["--simulation", "--config", str(config_path), "--label", "unknown"])


def test_timeline_annotations(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["simulation"]["timeline"] = [[0.05, "empty"], [0.1, "motion"]]
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (path,) = load_script("record").main(["--simulation", "--duration-s", "0.2", "--config", str(config_path)])
    with SessionReader(path) as session:
        assert session.annotations() == [
            (0, 5000, "breathing"), (5000, 5000, "empty"), (10000, session.n_samples - 10000, "motion"),
        ]
