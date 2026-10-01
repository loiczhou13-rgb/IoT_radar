"""HDF5 session files (schema v1): writer, reader, robustness."""

from __future__ import annotations

import datetime as dt
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest

from iot_radar.acquisition.recording import (
    LABELS,
    RADAR_ATTRIBUTES,
    SCHEMA_VERSION,
    SessionReader,
    SessionWriter,
    next_session_id,
    session_file_name,
)

RADAR = {
    "sample_rate_hz": 100e3,
    "center_frequency_hz": 3.5e9,
    "tx_waveform": "cw_offset",
    "tx_offset_hz": 244.140625,
    "tx_gain_db": -20.0,
    "rx_gain_db": 45.0,
    "rx_gain_mode": "manual",
    "n_rx_channels": 1,
    "channel_layout": "rx0",
    "firmware_version": "simulation",
    "source_kind": "simulation",
    "iq_stage": "raw_adc",
}


def _blocks(n_blocks: int = 4, n: int = 1000, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        (rng.integers(-2048, 2048, (1, n)) + 1j * rng.integers(-2048, 2048, (1, n))).astype(np.complex64)
        for _ in range(n_blocks)
    ]


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "session.h5"
    blocks = _blocks()
    start = dt.datetime(2026, 10, 5, 14, 30, 12, tzinfo=dt.timezone.utc)
    with SessionWriter(path, RADAR, scene={"room": "salle", "distance_m": 2.5},
                       config_yaml="sdr: {}\n", start_time_utc=start, with_ground_truth=True) as writer:
        for k, block in enumerate(blocks):
            writer.write_block(block, host_time_s=100.0 + 0.01 * k, overflow=(k == 2))
        writer.add_annotation(0, 4000, "breathing")
        writer.set_ground_truth(np.arange(3) * 0.1, np.array([0.0, 1e-3, 2e-3]))

    assert not os.access(path, os.W_OK)  # read-only once closed
    with SessionReader(path) as reader:
        attrs = reader.attributes
        assert attrs["schema_version"] == SCHEMA_VERSION
        assert all(attrs[key] == RADAR[key] for key in RADAR_ATTRIBUTES)
        assert attrs["room"] == "salle" and attrs["distance_m"] == 2.5
        assert attrs["subject_id"] == "" and math.isnan(attrs["obstacle_thickness_cm"])
        assert attrs["start_time_utc"] == start.isoformat() and attrs["config_yaml"] == "sdr: {}\n"
        assert (reader.n_channels, reader.n_samples, reader.sample_rate_hz) == (1, 4000, 100e3)
        np.testing.assert_array_equal(reader.read_iq(0, 4000), np.concatenate(blocks, axis=1))
        table = reader.blocks()
        assert table["sample_start"].tolist() == [0, 1000, 2000, 3000]
        np.testing.assert_allclose(table["host_time_s"], [0.0, 0.01, 0.02, 0.03])
        assert table["overflow"].tolist() == [0, 0, 1, 0]
        assert reader.annotations() == [(0, 4000, "breathing")]
        time_s, displacement_m = reader.ground_truth()
        np.testing.assert_array_equal(displacement_m, [0.0, 1e-3, 2e-3])
    with h5py.File(path, "r") as f:
        assert f["iq"].dtype == np.int16 and f["iq"].shape == (1, 4000, 2)
        assert f["iq"].chunks[1] == 65536 and f["iq"].maxshape == (1, None, 2)


def test_scaled_storage_of_non_adc_data(tmp_path: Path) -> None:
    path = tmp_path / "scaled.h5"
    x = (np.linspace(-1, 1, 500) * (1 + 0.5j)).astype(np.complex64)[np.newaxis, :]
    with SessionWriter(path, RADAR, iq_scale=1.0 / 30000) as writer:
        writer.write_block(x, 0.0, False)
    with SessionReader(path) as reader:
        assert reader.iq_scale == pytest.approx(1 / 30000)
        np.testing.assert_allclose(reader.read_iq(0, 500), x, atol=1 / 30000)


def test_never_overwrites_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "session.h5"
    SessionWriter(path, RADAR).close()
    with pytest.raises(FileExistsError):
        SessionWriter(path, RADAR)
    with pytest.raises(ValueError):
        SessionWriter(tmp_path / "x.h5", {k: v for k, v in RADAR.items() if k != "tx_gain_db"})
    with pytest.raises(ValueError):
        SessionWriter(tmp_path / "y.h5", RADAR, scene={"weather": "rain"})
    with SessionWriter(tmp_path / "z.h5", RADAR) as writer:
        with pytest.raises(ValueError):
            writer.add_annotation(0, 1, "sleeping")
        with pytest.raises(ValueError):
            writer.write_block(np.full((1, 4), 40000, np.complex64), 0.0, False)  # beyond int16
    assert set(LABELS) == {"empty", "breathing", "apnea", "motion", "unknown"}


def test_session_survives_a_killed_writer(tmp_path: Path) -> None:
    """Data flushed before the process dies stay readable (SWMR file)."""
    path = tmp_path / "killed.h5"
    script = textwrap.dedent(f"""
        import os, numpy as np
        from pathlib import Path
        from iot_radar.acquisition.recording import SessionWriter
        writer = SessionWriter(Path({str(path)!r}), {RADAR!r}, flush_interval_s=0.0)
        for k in range(5):
            writer.write_block(np.full((1, 1000), k + 1j, np.complex64), float(k), False)
        os._exit(0)   # killed: close() never runs
    """)
    subprocess.run([sys.executable, "-c", script], check=True)
    with SessionReader(path) as reader:
        assert reader.n_samples == 5000
        assert reader.read_iq(4999, 5000)[0, 0] == 4 + 1j
        assert len(reader.blocks()) == 5 and reader.annotations() == []


def test_file_names(tmp_path: Path) -> None:
    start = dt.datetime(2026, 10, 5, 14, 30, 12, tzinfo=dt.timezone.utc)
    assert session_file_name(42, start) == "session_0042_20261005_143012.h5"
    assert next_session_id(tmp_path / "missing") == 1
    (tmp_path / "session_0007_20261005_143012.h5").touch()
    (tmp_path / "notes.txt").touch()
    assert next_session_id(tmp_path) == 8
