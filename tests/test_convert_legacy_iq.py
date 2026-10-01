"""scripts/convert_legacy_iq.py on synthetic legacy recordings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from iot_radar.acquisition.recording import SessionReader
from iot_radar.acquisition.sources import ReplaySource
from script_loader import load_script


def _legacy_recording(folder: Path, index: int, label: int) -> np.ndarray:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(index)
    iq = (rng.normal(0, 3.0, 5000) + 1j * rng.normal(0, 3.0, 5000)).astype(np.complex64)
    iq.tofile(folder / f"{index}.iq")
    (folder / f"{index}.json").write_text(json.dumps({
        "f_s_dec_hz": 2000.0, "label": label, "env": "salle", "subset": "train",
        "sample_index": index, "utc_finished": "2026-05-15T07:27:54.662641+00:00",
        "duration_recorded_wall_s": 119.95,
    }), encoding="utf-8")
    return iq


def test_conversion(tmp_path: Path) -> None:
    convert = load_script("convert_legacy_iq")
    legacy = tmp_path / "old" / "train"
    iq_1 = _legacy_recording(legacy, 1, label=0)
    _legacy_recording(legacy, 2, label=1)
    (legacy / "3.iq").write_bytes(b"")  # no .json: skipped
    output = tmp_path / "sessions"

    paths = convert.main([str(tmp_path / "old"), "--output-dir", str(output)])
    assert [p.name for p in paths] == ["session_0001_20260515_072554.h5", "session_0002_20260515_072554.h5"]
    with SessionReader(paths[0]) as session:
        attrs = session.attributes
        assert attrs["sample_rate_hz"] == 2000.0 and attrs["iq_stage"] == "decimated_clutter_filtered"
        assert attrs["tx_offset_hz"] == pytest.approx(488.28125) and attrs["room"] == "salle"
        assert attrs["legacy_source_file"].endswith("old/train/1.iq")
        assert session.annotations() == [(0, 5000, "empty")]
        np.testing.assert_allclose(session.read_iq(0, 5000)[0], iq_1, atol=session.iq_scale)
    with SessionReader(paths[1]) as session:
        assert session.annotations()[0][2] == "breathing"

    assert convert.main([str(tmp_path / "old"), "--output-dir", str(output)]) == []  # already done
    source = ReplaySource(paths[0], max_block_samples=1000)
    assert source.read_block().samples.shape == (1, 1000)  # one stored block, cut for the replay
    source.close()
