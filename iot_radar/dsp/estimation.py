"""Spectra of the displacement and breath-by-breath analysis.

Place in the chain: after the demodulation.  The rate is the peak of
:func:`periodogram` (or of :func:`weighted_lomb_scargle` with the M3
weights), located by :func:`iot_radar.dsp.detection.breathing_metrics`;
:func:`breath_cycles` finds the individual breaths (apnea alert).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, get_window


@dataclass
class Spectrum:
    """One-sided power spectrum: frequencies (Hz), power, resolution before zero-padding (Hz)."""

    f_hz: np.ndarray
    psd: np.ndarray
    native_resolution_hz: float


def periodogram(
    x: np.ndarray,
    f_s_hz: float,
    window_name: str = "hann",
    n_fft: int | None = None,
) -> Spectrum:
    """Windowed one-sided periodogram ``|FFT|² / Σ window²`` of a real signal (mean removed)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    window = get_window(window_name, n) if window_name != "none" else np.ones(n)
    n_fft = max(int(n_fft or n), n)
    spectrum = np.fft.rfft((x - x.mean()) * window, n=n_fft)
    psd = (np.abs(spectrum) ** 2) / max(np.sum(window * window), 1e-30)
    f_hz = np.fft.rfftfreq(n_fft, d=1.0 / f_s_hz)
    return Spectrum(f_hz, psd, f_s_hz / max(n, 1))


def weighted_lomb_scargle(
    t_s: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    f_hz: np.ndarray,
    window_name: str = "hann",
) -> Spectrum:
    """Weighted Lomb-Scargle periodogram with a floating mean (tolerates gaps), power in [0, 1].

    For each frequency, ``A·cos + B·sin + c`` is fitted by weighted least
    squares; the power is the χ² reduction over the weighted variance.
    Samples of weight 0 do not count, so contaminated stretches are skipped
    (M3).  The taper *window_name* is multiplied into the weights, for the
    same leakage as :func:`periodogram`.  M. Zechmeister and M. Kürster,
    "The generalised Lomb-Scargle periodogram", A&A 496, 577–584, 2009.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if window_name != "none":
        w = w * get_window(window_name, x.size)
    f_hz = np.asarray(f_hz, dtype=np.float64)
    duration_s = float(t_s[-1] - t_s[0]) if t_s.size > 1 else 1.0
    if w.sum() <= 0.0:
        return Spectrum(f_hz, np.zeros_like(f_hz), 1.0 / duration_s)
    w = w / w.sum()

    y_mean = np.sum(w * x)
    y = x - y_mean
    yy = np.sum(w * y * y)
    omega_t = 2.0 * np.pi * np.outer(f_hz, t_s)                # (n_f, N)
    cos, sin = np.cos(omega_t), np.sin(omega_t)
    c, s = cos @ w, sin @ w
    cc = (cos * cos) @ w - c * c
    ss = (sin * sin) @ w - s * s
    cs = (cos * sin) @ w - c * s
    yc, ys = cos @ (w * y), sin @ (w * y)
    determinant = cc * ss - cs * cs
    power = np.zeros_like(f_hz)
    valid = (determinant > 1e-12) & (yy > 0.0)
    power[valid] = ((ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys)[valid]
                    / (determinant[valid] * yy))
    return Spectrum(f_hz, power, 1.0 / duration_s)


def remove_segment_offsets(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Remove the weighted mean of each run of positive weights separately.

    A gap cut around an anomaly may hide a 2π phase slip (λ/2 step): each
    segment gets its own offset.
    """
    x = np.array(x, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    edges = np.flatnonzero(np.diff(np.r_[0, (w > 0).astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        x[start:stop] -= np.sum(w[start:stop] * x[start:stop]) / np.sum(w[start:stop])
    return x


@dataclass
class BreathCycles:
    """Breaths of the band-passed displacement, seen as chest expansion (``−d``).

    Peaks = ends of inhalation, troughs = ends of exhalation.  ``rate_hz``
    from the mean interval (``None`` below 2 breaths or out of band);
    ``variability`` = std / mean of the intervals; ``depth`` = median
    peak-to-trough; ``ie_ratio`` = inhalation / exhalation duration;
    ``since_last_s`` = time from the last peak to the end of the window.
    """

    peak_indices: np.ndarray
    trough_indices: np.ndarray
    intervals_s: np.ndarray
    rate_hz: float | None
    variability: float
    depth: float
    ie_ratio: float
    since_last_s: float


def breath_cycles(
    x: np.ndarray,
    f_s_hz: float,
    band_hz: tuple[float, float],
    prominence_std: float = 0.5,
) -> BreathCycles:
    """Breaths of *x*: peaks at least ``1 / high`` apart, prominence ≥ *prominence_std* std."""
    chest = -np.asarray(x, dtype=np.float64)
    nan = float("nan")
    empty = np.zeros(0, dtype=int)
    if chest.size < 4 or chest.std() == 0:
        return BreathCycles(empty, empty, np.zeros(0), None, nan, nan, nan, chest.size / f_s_hz)
    peak_options = dict(distance=max(1, int(f_s_hz / band_hz[1])), prominence=prominence_std * chest.std())
    peaks, _ = find_peaks(chest, **peak_options)
    troughs, _ = find_peaks(-chest, **peak_options)

    intervals_s = np.diff(peaks) / f_s_hz
    rate_hz = None
    if intervals_s.size:
        mean_rate_hz = float(1.0 / intervals_s.mean())
        rate_hz = mean_rate_hz if band_hz[0] <= mean_rate_hz <= band_hz[1] else None
    variability = float(intervals_s.std() / intervals_s.mean()) if intervals_s.size >= 2 else nan

    depths, inhalation_lengths, exhalation_lengths = [], [], []
    for peak in peaks:
        before, after = troughs[troughs < peak], troughs[troughs > peak]
        if before.size and after.size:
            depths.append(chest[peak] - 0.5 * (chest[before[-1]] + chest[after[0]]))
            inhalation_lengths.append(peak - before[-1])
            exhalation_lengths.append(after[0] - peak)
    depth = float(np.median(depths)) if depths else nan
    ie_ratio = float(np.median(inhalation_lengths) / np.median(exhalation_lengths)) if inhalation_lengths else nan
    since_last_s = float((chest.size - 1 - peaks[-1]) / f_s_hz) if peaks.size else chest.size / f_s_hz
    return BreathCycles(peaks, troughs, intervals_s, rate_hz, variability, depth, ie_ratio, since_last_s)
