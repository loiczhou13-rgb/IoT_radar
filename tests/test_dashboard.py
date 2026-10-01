"""Headless smoke tests of the matplotlib dashboard (Agg backend)."""

from __future__ import annotations

import logging

import pytest

from characterization_cases import pipeline_config, seeded_default_rng
from iot_radar.acquisition.sources import open_source
from iot_radar.pipeline import build_context, streaming_frame_generator
from iot_radar.ui.dashboard import DashboardRadar


@pytest.fixture(scope="module")
def frame() -> dict:
    logging.disable(logging.WARNING)
    with seeded_default_rng():
        frames = streaming_frame_generator(pipeline_config(), open_source(pipeline_config(), simulation=True))
        first = next(frames)
        frames.close()
    logging.disable(logging.NOTSET)
    return first


@pytest.mark.parametrize("show_score", [True, False])
def test_dashboard_updates(frame: dict, show_score: bool) -> None:
    cfg = pipeline_config()
    dashboard = DashboardRadar(config=cfg, context=build_context(cfg), show_presence_score=show_score)
    artists = dashboard.update_frame(None)
    assert len(artists) == (4 if show_score else 3)
    dashboard.update_frame(frame)
    status = dashboard._status_text.get_text()
    if show_score:
        assert "DÉTECTÉE" in status or "Aucune" in status
    else:
        assert status == f"Trame {frame['n_trame']}"


def test_info_box_live_and_replay(frame: dict) -> None:
    cfg = pipeline_config()
    live = DashboardRadar(config=cfg, context=build_context(cfg))
    replay = DashboardRadar(config=cfg, context=build_context(cfg), show_presence_score=False, title="Replay")
    live_lines = live._info_text.get_text().splitlines()
    replay_lines = replay._info_text.get_text().splitlines()
    assert live_lines[:len(replay_lines)] == replay_lines
    assert "─────── LIVE ───────" in live_lines and "─────── LIVE ───────" not in replay_lines
    assert replay._fig._suptitle.get_text() == "Replay"
