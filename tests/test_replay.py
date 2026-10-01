"""ReplaySource: a recorded session replays exactly like the live stream."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from characterization_cases import pipeline_config
from iot_radar.acquisition.recording import SessionWriter
from iot_radar.acquisition.sources import CWSimulationSource, ReplaySource
from iot_radar.pipeline import streaming_frame_generator
from test_sessions import RADAR


def _simulation() -> CWSimulationSource:
    return CWSimulationSource(
        f_c_hz=3.5e9, f_s_hz=100e3, buffer_size=4096, tx_offset_hz=244.140625,
        breath_rate_hz=0.3, breath_amplitude_mm=10, seed=11,
    )


def test_replayed_session_gives_the_live_pipeline_output(tmp_path: Path) -> None:
    logging.disable(logging.WARNING)
    cfg = pipeline_config()
    live = [f for _, f in zip(range(4), streaming_frame_generator(cfg, _simulation()))]

    path = tmp_path / "session.h5"
    source = _simulation()
    with SessionWriter(path, RADAR) as writer:
        for _ in range(160):  # 160 x 4096 samples = 6.6 s: warm-up + 4 frames
            block = source.read_block()
            writer.write_block(block.samples, block.host_time_s, block.overflow)
    replayed = list(streaming_frame_generator(cfg, ReplaySource(path)))
    logging.disable(logging.NOTSET)

    assert len(replayed) >= 4
    for live_frame, replay_frame in zip(live, replayed[:4]):
        np.testing.assert_array_equal(live_frame["spectrum_column_db"], replay_frame["spectrum_column_db"])
        assert live_frame["presence_score"] == replay_frame["presence_score"]


def test_blocks_flags_and_long_blocks(tmp_path: Path) -> None:
    path = tmp_path / "session.h5"
    with SessionWriter(path, RADAR) as writer:
        writer.write_block(np.ones((1, 10), np.complex64), 5.0, False)
        writer.write_block(np.full((1, 25), 2, np.complex64), 6.0, True)
    source = ReplaySource(path, max_block_samples=10)
    pieces = []
    while (block := source.read_block()) is not None:
        pieces.append((block.sample_start, block.samples.shape[1], round(block.host_time_s, 6), block.overflow))
    source.close()
    assert source.sample_rate_hz == 100e3 and source.kind == "replay"
    assert pieces == [(0, 10, 0.0, False), (10, 10, 1.0, True), (20, 10, 1.0001, False), (30, 5, 1.0002, False)]


def test_replayed_session_gives_the_live_phase_pipeline_output(tmp_path: Path) -> None:
    from iot_radar.pipeline import VitalSignsPipeline

    cfg = pipeline_config()
    cfg["sdr"]["center_frequency_hz"] = 3.5e9

    def outputs(source):
        pipeline = VitalSignsPipeline.from_config(cfg, source.sample_rate_hz, 244.140625)
        return [o for o in pipeline.run(source)]

    logging.disable(logging.WARNING)
    live_source = _simulation()
    live = []
    pipeline = VitalSignsPipeline.from_config(cfg, live_source.sample_rate_hz, 244.140625)
    for _ in range(600):  # 600 x 4096 samples = 24.6 s
        live += pipeline.process_block(live_source.read_block())

    path = tmp_path / "phase.h5"
    source = _simulation()
    with SessionWriter(path, RADAR) as writer:
        for _ in range(600):
            block = source.read_block()
            writer.write_block(block.samples, block.host_time_s, block.overflow)
    replayed = outputs(ReplaySource(path))
    logging.disable(logging.NOTSET)

    assert len(replayed) == len(live) > 40 and live[-1].analysis is not None
    for a, b in zip(live, replayed):
        assert a.t_s == b.t_s and a.breathing.state == b.breathing.state
        assert a.breathing.confidence == b.breathing.confidence
