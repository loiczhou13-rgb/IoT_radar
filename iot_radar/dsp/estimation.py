"""Breathing-rate estimators for a band-limited displacement waveform.

Place in the chain: after the phase demodulation and the band-pass filter.
All functions take a **real** waveform ``x`` (typically the band-passed chest
displacement) sampled at ``f_s_hz`` and a search band ``(low, high)`` in Hz.
They return a rate in **Hz** (× 60 for breaths per minute) or ``None`` when
no estimate is possible.

The breathing rate used by the detector is the peak of the displacement
spectrum (:func:`periodogram`, or :func:`weighted_lomb_scargle` with the
contamination weights); the peak itself is located by
:func:`iot_radar.dsp.detection.breathing_metrics`.  :func:`breath_cycles`
analyses the individual breaths (intervals, depth, time since the last
breath, which drives the apnea alert).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, get_window


@dataclass
class Spectrum:
    """One-sided power spectrum of a real window.

    Attributes
    ----------
    f_hz : numpy.ndarray
        Frequency axis (Hz), from 0 to ``f_s / 2``.
    psd : numpy.ndarray
        Periodogram (linear power, arbitrary units), same shape as ``f_hz``.
    native_resolution_hz : float
        Resolution ``f_s / N`` of the window before zero-padding (Hz).
    """

    f_hz: np.ndarray
    psd: np.ndarray
    native_resolution_hz: float


def periodogram(
    x: np.ndarray,
    f_s_hz: float,
    window_name: str = "hann",
    n_fft: int | None = None,
) -> Spectrum:
    """Windowed one-sided periodogram, optionally zero-padded.

    Parameters
    ----------
    x : numpy.ndarray
        Real signal, 1-D (its mean is removed).
    f_s_hz : float
        Sampling rate (Hz).
    window_name : str, optional
        ``scipy.signal.get_window`` name, or ``"none"``.
    n_fft : int or None, optional
        FFT size (≥ ``len(x)``); default ``len(x)``.

    Returns
    -------
    Spectrum
        ``|FFT|² / Σ window²`` on ``n_fft // 2 + 1`` frequencies.
    """
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
    """Weighted Lomb-Scargle periodogram with a floating mean (tolerant to gaps).

    For every frequency ``f``, the model ``A·cos(2πft) + B·sin(2πft) + c`` is
    fitted to *x* by weighted least squares; the power is the reduction of
    the weighted χ² it brings, divided by the weighted variance of *x*
    (J. T. VanderPlas, "Understanding the Lomb-Scargle periodogram", ApJS
    236(1), 2018, §5–6: generalised periodogram with per-point weights).
    Samples of weight 0 do not count at all, so contaminated segments of a
    window are skipped instead of suspending the measurement (M3,
    MOTION_ROBUSTNESS_PLAN.md §3.4).

    Parameters
    ----------
    t_s : numpy.ndarray
        Sample times (s), shape ``(N,)``; need not be uniform.
    x : numpy.ndarray
        Real signal (e.g. the detrended chest displacement), shape ``(N,)``.
    weights : numpy.ndarray
        Non-negative weight of each sample, shape ``(N,)``.
    f_hz : numpy.ndarray
        Frequencies to evaluate (Hz, > 0).
    window_name : str, optional
        Taper multiplied into the weights (``"hann"`` like
        :func:`periodogram`, or ``"none"``), to keep the same spectral leakage
        as the unweighted chain.

    Returns
    -------
    Spectrum
        Normalised power in [0, 1] on *f_hz*; ``native_resolution_hz`` is
        ``1 / duration``.
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
    """Remove the weighted mean of every run of positive weights separately.

    A gap (weight 0) cut around an in-band anomaly may hide a step: when a
    passer-by's echo crosses the breathing band, the phasor around the arc
    centre can turn once around it and the unwrapped phase slips by 2π, i.e.
    λ/2 of displacement.  Each segment between gaps then gets its own offset,
    as the floating mean of the generalised Lomb-Scargle periodogram does for
    a single segment (J. T. VanderPlas, ApJS 236(1), 2018, §6).

    Parameters
    ----------
    x : numpy.ndarray
        Real signal, shape ``(N,)``.
    weights : numpy.ndarray
        Non-negative weights, shape ``(N,)``; samples of weight 0 are left
        unchanged.

    Returns
    -------
    numpy.ndarray
        Copy of *x* with a zero weighted mean on every segment.
    """
    x = np.array(x, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    edges = np.flatnonzero(np.diff(np.r_[0, (w > 0).astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        x[start:stop] -= np.sum(w[start:stop] * x[start:stop]) / np.sum(w[start:stop])
    return x


@dataclass
class BreathCycles:
    """Breath-by-breath analysis of the band-passed displacement.

    The waveform is analysed as *chest expansion towards the radar*
    (``chest = −d``: an inhalation brings the chest closer and decreases the
    range), so the peaks are ends of inhalation and the troughs ends of
    exhalation.

    Attributes
    ----------
    peak_indices, trough_indices : numpy.ndarray
        Sample indices of the end-of-inhalation peaks / end-of-exhalation troughs.
    intervals_s : numpy.ndarray
        Intervals between successive peaks (s).
    rate_hz : float or None
        Rate from the mean interval (``None`` with fewer than two breaths or
        out of band).
    variability : float
        Coefficient of variation of the intervals (std / mean); NaN with
        fewer than 3 breaths.
    depth : float
        Median peak-to-trough excursion (unit of the waveform).
    ie_ratio : float
        Median inhalation / exhalation duration ratio (typically 0.5–0.8 at rest).
    since_last_s : float
        Time from the last inhalation peak to the end of the window (s).
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
    """Detect the individual breaths of a band-passed displacement waveform *x*.

    Peaks closer than ``1 / high`` are merged and must have a prominence of at
    least ``prominence_std`` standard deviations.
    """
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
