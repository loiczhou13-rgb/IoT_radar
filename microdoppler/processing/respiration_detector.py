from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .microdoppler_features import MicroDopplerFeatures
from .respiration_periodicity import PeriodicityEstimate, estimate_periodicity_fft


@dataclass
class DetectorState:
    feature_history: list[float]


@dataclass(frozen=True)
class RespirationDecision:
    is_present: bool
    confidence: float
    rate_hz: float
    rate_bpm: float
    snr_db: float


def init_state() -> DetectorState:
    return DetectorState(feature_history=[])


def update_detector(
    *,
    state: DetectorState,
    feature: MicroDopplerFeatures,
    dt_feature_s: float,
    min_resp_hz: float,
    max_resp_hz: float,
    min_stable_seconds: float,
    snr_db_min: float,
    confidence_min: float,
) -> tuple[RespirationDecision, Optional[PeriodicityEstimate]]:
    state.feature_history.append(float(feature.band_energy))

    max_len = int(max(32, round(min_stable_seconds / max(dt_feature_s, 1e-6))))
    if len(state.feature_history) > max_len:
        state.feature_history = state.feature_history[-max_len:]

    est: Optional[PeriodicityEstimate] = None
    conf = 0.0
    rate_hz = 0.0
    rate_bpm = 0.0

    if len(state.feature_history) >= max_len:
        est = estimate_periodicity_fft(
            np.asarray(state.feature_history, dtype=np.float32),
            dt_s=dt_feature_s,
            min_hz=min_resp_hz,
            max_hz=max_resp_hz,
        )
        conf = float(est.confidence)
        rate_hz = float(est.rate_hz)
        rate_bpm = float(est.rate_bpm)

    # Decision rule (simple, maintainable)
    is_present = (feature.band_snr_db >= snr_db_min) and (conf >= confidence_min) and (rate_hz > 0.0)
    # Combine confidence with SNR margin (soft)
    snr_margin = float(np.clip((feature.band_snr_db - snr_db_min) / max(12.0, snr_db_min), 0.0, 1.0))
    combined_conf = float(np.clip(0.65 * conf + 0.35 * snr_margin, 0.0, 1.0))

    decision = RespirationDecision(
        is_present=bool(is_present),
        confidence=combined_conf,
        rate_hz=rate_hz,
        rate_bpm=rate_bpm,
        snr_db=float(feature.band_snr_db),
    )
    return decision, est

