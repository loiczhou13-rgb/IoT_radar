"""Headless smoke test of the phase dashboard (Agg backend)."""

from __future__ import annotations

import dataclasses

from iot_radar.ui.dashboard import PhaseDashboard
from script_loader import load_script
from test_vital_signs_pipeline import _config, _run


def test_dashboard_shows_the_pipeline_outputs() -> None:
    cfg = _config(timeline=[[0, "breathing"], [30, "apnea"]], heart_amplitude_mm=0.15)
    outputs = _run(cfg, 42.0)
    run_radar = load_script("run_radar")

    from iot_radar.pipeline import VitalSignsPipeline

    pipeline = VitalSignsPipeline.from_config(cfg, 100e3, 244.140625)
    context = run_radar.dashboard_context(cfg, pipeline, 244.140625, "Test", "simulation")
    dashboard = PhaseDashboard(context)
    for output in outputs:
        dashboard.update(output)
    assert dashboard._state_text.get_text() == "BREATHING DETECTED"
    assert dashboard._alert_text.get_visible() and "NO BREATH" in dashboard._alert_text.get_text()
    assert "Reported rate" in dashboard._live_text.get_text()
    assert dashboard._iq_circle.get_visible()

    restarted = dataclasses.replace(outputs[-1], discontinuity=True, t_s=outputs[-1].t_s + 0.5,
                                    breathing=dataclasses.replace(outputs[-1].breathing, apnea=False))
    dashboard.update(restarted)
    assert dashboard._alert_text.get_visible() and "SAMPLES LOST" in dashboard._alert_text.get_text()
