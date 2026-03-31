from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicityEstimate:
    rate_hz: float
    rate_bpm: float
    confidence: float


def estimate_periodicity_fft(
    series: np.ndarray,
    *,
    dt_s: float,
    min_hz: float,
    max_hz: float,
) -> PeriodicityEstimate:
    """
    Estimate periodicity using FFT peak picking on a 1D feature time-series.
    Returns a confidence score based on peak prominence vs median.
    """
    x = np.asarray(series, dtype=np.float32)
    if x.size < 16:
        return PeriodicityEstimate(rate_hz=0.0, rate_bpm=0.0, confidence=0.0)

    x = x - float(np.mean(x))
    x = x * np.hanning(x.size).astype(np.float32)

    n = int(2 ** int(np.ceil(np.log2(x.size))))
    X = np.fft.rfft(x, n=n)
    f = np.fft.rfftfreq(n, d=dt_s).astype(np.float32)

    mask = (f >= float(min_hz)) & (f <= float(max_hz))
    if not np.any(mask):
        return PeriodicityEstimate(rate_hz=0.0, rate_bpm=0.0, confidence=0.0)

    mag = np.abs(X).astype(np.float32)
    mag_band = mag[mask]
    f_band = f[mask]

    k = int(np.argmax(mag_band))
    peak = float(mag_band[k])
    rate_hz = float(f_band[k])

    baseline = float(np.median(mag_band) + 1e-12)
    confidence = float(np.clip((peak - baseline) / max(peak, 1e-12), 0.0, 1.0))
    return PeriodicityEstimate(rate_hz=rate_hz, rate_bpm=60.0 * rate_hz, confidence=confidence)

