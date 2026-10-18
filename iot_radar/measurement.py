"""Measurement mode (M4): one decision over a silence started by the operator.

The rescuers stand still out of the beam; the next ``duration_s`` seconds of
slow time (and their M3 weights) are analysed once, with the same processing
as the continuous mode, while the latter keeps running.  This module holds
the settings, the results and the decision rule; the pipeline accumulates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iot_radar.config import check_keys

DECISION_BREATHING = "breathing"
DECISION_NO_BREATHING = "no_breathing"
DECISION_INCONCLUSIVE = "inconclusive"

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_DONE = "done"


@dataclass
class MeasurementSettings:
    """``measurement`` section: length (s), and minimum share of samples with a
    weight ≥ ``contaminated_below`` (M3) for a conclusive decision."""

    duration_s: float = 30.0
    min_valid_fraction: float = 0.5

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "MeasurementSettings":
        """From the YAML section (defaults for missing keys, unknown keys refused)."""
        check_keys("measurement", section, ["duration_s", "min_valid_fraction"])
        d = cls()
        return cls(
            duration_s=float(section.get("duration_s", d.duration_s)),
            min_valid_fraction=float(section.get("min_valid_fraction", d.min_valid_fraction)),
        )


@dataclass
class MeasurementResult:
    """Outcome of one measurement.

    Times on the slow-time axis of the outputs (s).  ``decision``:
    ``"breathing"``, ``"no_breathing"`` or ``"inconclusive"`` with its
    ``reasons``.  Fractions are shares of the samples (contaminated: weight
    below ``contaminated_below``; activity: someone moving around).
    ``analysis`` is the window analysis (``None`` if cut short).
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
    """Measurement mode in one output: ``"idle"``, ``"running"`` or ``"done"`` (only the
    output that completes it), time measured / length (s), last result."""

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
    """Decision and reasons of an inconclusive one.

    Inconclusive whatever the confidence on saturation, target motion,
    discontinuity or a valid share below *min_valid_fraction*; otherwise
    breathing ≥ *threshold_on*, no breathing < *threshold_off*, inconclusive
    in between (same hysteresis as the continuous detector).
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
