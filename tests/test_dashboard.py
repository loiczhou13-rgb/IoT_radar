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


def test_dashboard_shows_the_motion_robustness_and_the_measurement() -> None:
    """M3 indicators, pause cause, key m and the result of a measurement (M4)."""
    from iot_radar.measurement import MeasurementResult, MeasurementStatus

    cfg = _config()
    outputs = _run(cfg, 26.0)
    run_radar = load_script("run_radar")
    from iot_radar.pipeline import VitalSignsPipeline

    pipeline = VitalSignsPipeline.from_config(cfg, 100e3, 244.140625)
    context = run_radar.dashboard_context(cfg, pipeline, 244.140625, "Test", "simulation")
    assert "Motion robustness" in [title for title, _ in context["parameters"]]
    requests = []
    dashboard = PhaseDashboard(context, on_measure=lambda: requests.append(True))
    assert dashboard._measurement_text.get_text() == "press m to measure"
    dashboard._on_key(type("KeyEvent", (), {"key": "m"})())
    assert requests == [True] and "requested" in dashboard._measurement_text.get_text()

    last = outputs[-1]
    assert last.contamination is not None
    running = dataclasses.replace(last, measurement=MeasurementStatus("running", 12.0, 30.0, None))
    dashboard.update(running)
    assert "MEASURING 12 / 30 s" in dashboard._measurement_text.get_text()
    assert "Current weight" in dashboard._live_text.get_text()

    result = MeasurementResult(10.0, 40.0, "breathing", (), 18.0, 0.93, 0.98, 0.05, 0.0, 0.2, False, False)
    done = dataclasses.replace(last, t_s=last.t_s + 0.5, measurement=MeasurementStatus("done", 30.0, 30.0, result))
    dashboard.update(done)
    assert dashboard._measurement_text.get_text() == "Measurement: BREATHING 18.0 /min"
    assert "Last measurement" in dashboard._live_text.get_text()

    saturated = dataclasses.replace(last, t_s=last.t_s + 1.0, breathing=dataclasses.replace(
        last.breathing, state="MOTION", pause_reason="saturation"))
    dashboard.update(saturated)
    assert dashboard._state_text.get_text() == "ADC SATURATION — PAUSED"
