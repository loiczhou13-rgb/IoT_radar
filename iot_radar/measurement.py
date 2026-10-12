"""Operator-triggered measurement over a silence window (M4).

Place in the chain: on top of the continuous processing of
:class:`iot_radar.pipeline.VitalSignsPipeline`, which keeps running.  In the
field, the rescuers stand still and leave the beam, then the operator starts
a measurement: the slow-time samples of the next ``duration_s`` seconds (30 s
by default) and their M3 weights are accumulated, then analysed once, with
the same M1 to M3 processing as the continuous mode.  The result is a
decision, a breathing rate, a confidence and a quality summary
(MOTION_ROBUSTNESS_PLAN.md §3.5).

This module holds the settings, the result structures and the decision rule;
the pipeline does the accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DECISION_BREATHING = "breathing"
DECISION_NO_BREATHING = "no_breathing"
DECISION_INCONCLUSIVE = "inconclusive"

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_DONE = "done"


@dataclass
class MeasurementSettings:
    """Measurement mode (``measurement`` section of the configuration).

    Attributes
    ----------
    duration_s : float
        Length of a measurement (s).
    min_valid_fraction : float
        Minimum share of the measurement with a weight of at least
        ``contaminated_below`` (M3) for a conclusive decision.
    """

    duration_s: float = 30.0
    min_valid_fraction: float = 0.5

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "MeasurementSettings":
        """Build the settings from the ``measurement`` section (defaults for missing keys)."""
        d = cls()
        return cls(
            duration_s=float(section.get("duration_s", d.duration_s)),
            min_valid_fraction=float(section.get("min_valid_fraction", d.min_valid_fraction)),
        )


@dataclass
class MeasurementResult:
    """Outcome of one measurement.

    Attributes
    ----------
    t_start_s, t_end_s : float
        Slow time of the first and last samples of the measurement (s since
        the start of the stream, on the time axis of the outputs).
    decision : str
        ``"breathing"``, ``"no_breathing"`` or ``"inconclusive"``.
    reasons : tuple of str
        Why the decision is inconclusive (``"saturation"``,
        ``"target_motion"``, ``"discontinuity"``, ``"too_contaminated"``,
        ``"uncertain"``); empty otherwise.
    rate_bpm : float or None
        Breathing rate (breaths per minute) when the decision is ``"breathing"``.
    confidence : float
        Confidence of the measurement window, multiplied by its mean weight (M3).
    mean_weight : float
        Mean M3 weight of the samples (1 without M3).
    contaminated_fraction : float
        Share of the samples whose weight is below ``contaminated_below``.
    saturated_fraction : float
        Share of the samples that come from saturated ADC blocks.
    activity_fraction : float
        Share of the samples during which someone moved around the radar
        (out-of-band activity, M3).
    target_motion : bool
        The target itself moved during the measurement.
    discontinuity : bool
        The stream had a discontinuity: the measurement was cut short.
    analysis : object or None
        The :class:`iot_radar.pipeline.WindowAnalysis` of the measurement
        window, for display (``None`` if it was cut short).
    """

    t_start_s: float
    t_end_s: float
    decision: str
    reasons: tuple[str, ...]
    rate_bpm: float | None
    confidence: float
    mean_weight: float
    contaminated_fraction: float
    saturated_fraction: float
    activity_fraction: float
    target_motion: bool
    discontinuity: bool
    analysis: Any = None


@dataclass
class MeasurementStatus:
    """State of the measurement mode in one output.

    Attributes
    ----------
    state : str
        ``"idle"``, ``"running"`` or ``"done"`` (the output that completes a
        measurement; the next ones are ``"idle"`` again).
    elapsed_s : float
        Time already measured (s).
    duration_s : float
        Length of the current (or last) measurement (s).
    result : MeasurementResult or None
        Result of the last completed measurement.
    """

    state: str
    elapsed_s: float
    duration_s: float
    result: MeasurementResult | None


def measurement_decision(
    confidence: float,
    valid_fraction: float,
    saturated: bool,
    target_motion: bool,
    discontinuity: bool,
    min_valid_fraction: float,
    threshold_on: float,
    threshold_off: float,
) -> tuple[str, tuple[str, ...]]:
    """Decision of a measurement and the reasons of an inconclusive one.

    ``"breathing"`` when the confidence reaches *threshold_on*,
    ``"no_breathing"`` below *threshold_off*, ``"inconclusive"`` in between,
    or whatever the confidence when the ADC saturated, the target moved, the
    stream was cut, or less than *min_valid_fraction* of the window is valid.
    The thresholds are those of the continuous detector (hysteresis band).
    """
    reasons = []
    if saturated:
        reasons.append("saturation")
    if target_motion:
        reasons.append("target_motion")
    if discontinuity:
        reasons.append("discontinuity")
    if valid_fraction < min_valid_fraction:
        reasons.append("too_contaminated")
    if reasons:
        return DECISION_INCONCLUSIVE, tuple(reasons)
    if confidence >= threshold_on:
        return DECISION_BREATHING, ()
    if confidence < threshold_off:
        return DECISION_NO_BREATHING, ()
    return DECISION_INCONCLUSIVE, ("uncertain",)
