"""Headless smoke tests of the matplotlib dashboard (Agg backend)."""

from __future__ import annotations

import logging

import pytest

from characterization_cases import pipeline_config, seeded_default_rng
from MicroDopplerDetection.main import _build_context, _streaming_frame_generator
from MicroDopplerDetection.utils.display import DashboardRadar


@pytest.fixture(scope="module")
def frame() -> dict:
    logging.disable(logging.WARNING)
    with seeded_default_rng():
        frames = _streaming_frame_generator(pipeline_config(), simulation=True)
        first = next(frames)
        frames.close()
    logging.disable(logging.NOTSET)
    return first


@pytest.mark.parametrize("show_score", [True, False])
def test_dashboard_updates(frame: dict, show_score: bool) -> None:
    cfg = pipeline_config()
    dashboard = DashboardRadar(config=cfg, context=_build_context(cfg), show_presence_score=show_score)
    artists = dashboard.update_frame(None)
    assert len(artists) == (4 if show_score else 3)
    dashboard.update_frame(frame)
    status = dashboard._status_text.get_text()
    if show_score:
        assert "DÉTECTÉE" in status or "Aucune" in status
    else:
        assert status == f"Trame {frame['n_trame']}"
