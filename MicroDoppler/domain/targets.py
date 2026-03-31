from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RespirationPreset:
    """
    Domain-level parameters. This file is meant to be edited by non-DSP contributors.
    """

    # Typical adult respiration: ~6–30 bpm => 0.1–0.5 Hz
    min_resp_hz: float = 0.10
    max_resp_hz: float = 0.50

    # For micro-doppler respiration we look very close to zero velocity but not exactly DC
    # (to avoid static clutter/DC leakage).
    band_v_min_ms: float = 0.01
    band_v_max_ms: float = 0.40

    min_stable_seconds: float = 6.0


RESPIRATION = RespirationPreset()

