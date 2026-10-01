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
