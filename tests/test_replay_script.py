"""scripts/replay.py: configuration of a replayed session."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from iot_radar.acquisition.recording import SessionWriter
from iot_radar.acquisition.sources import ReplaySource
from iot_radar.config import DEFAULT_RADAR_CONFIG, load_config
from script_loader import load_script
from test_sessions import RADAR


def test_replay_uses_the_stored_configuration(tmp_path: Path) -> None:
    replay = load_script("replay")
    stored = {"sdr": {"sample_rate_hz": 100e3}, "note": "stored"}
    with_config = tmp_path / "with.h5"
    with SessionWriter(with_config, RADAR, config_yaml=yaml.safe_dump(stored)) as writer:
        writer.write_block(np.zeros((1, 4), np.complex64), 0.0, False)
    without_config = tmp_path / "without.h5"
    SessionWriter(without_config, RADAR).close()

    source = ReplaySource(with_config)
    assert replay.replay_config(source, None) == stored
    assert replay.replay_config(source, str(DEFAULT_RADAR_CONFIG)) == load_config(DEFAULT_RADAR_CONFIG)
    source.close()
    source = ReplaySource(without_config)
    assert replay.replay_config(source, None) == load_config(DEFAULT_RADAR_CONFIG)
    source.close()


def test_replay_runs_the_phase_pipeline(tmp_path: Path) -> None:
    from characterization_cases import pipeline_config

    cfg = pipeline_config()
    cfg["recording"]["sessions_dir"] = str(tmp_path)
    cfg["logging"] = {"level": "INFO", "to_file": True}
    config_path = tmp_path / "radar.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (session,) = load_script("record").main(["--simulation", "--duration-s", "22", "--config", str(config_path),
                                             "--log-file", str(tmp_path / "record.log")])

    log_file = tmp_path / "replay.log"
    load_script("replay").main([str(session), "--speed", "0", "--headless", "--log-file", str(log_file)])
    lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if "confidence=" in line]
    assert len(lines) > 30 and "WARMUP" in lines[0]
    assert "WARMUP" not in lines[-1]  # the 20 s window is full at the end
