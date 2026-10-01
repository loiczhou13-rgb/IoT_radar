"""Breathing-rate estimators for a band-limited displacement waveform.

Place in the chain: after the phase demodulation and the band-pass filter.
All functions take a **real** waveform ``x`` (typically the band-passed chest
displacement) sampled at ``f_s_hz`` and a search band ``(low, high)`` in Hz.
They return a rate in **Hz** (× 60 for breaths per minute) or ``None`` when
no estimate is possible.

Four complementary estimators are provided, because they fail in different
ways:

* :func:`fft_peak_rate` — spectral peak; robust to noise, resolution
  ``1 / T_window`` (refined by zero-padding and parabolic interpolation);
* :func:`acf_rate` — first dominant autocorrelation lag; robust to
  non-sinusoidal breathing (harmonics);
* :func:`peak_count_rate` — mean interval between peaks; breath by breath,
  sensitive to residual noise;
* :func:`zero_crossing_rate` — zero-crossing count; cheap, needs a clean
  band-passed signal.

The detector uses the FFT peak; the others are cross-checks.
:func:`breath_cycles` analyses the individual breaths (intervals,
variability, depth, inhalation/exhalation ratio, time since the last breath).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, find_peaks, get_window


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


def _parabolic_offset(left: float, centre: float, right: float) -> float:
    """Sub-sample position of a peak from three samples (in [-0.5, 0.5])."""
    curvature = left - 2.0 * centre + right
    if curvature == 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))


def fft_peak_rate(
    x: np.ndarray,
    f_s_hz: float,
    band_hz: tuple[float, float],
    zero_pad_factor: int = 8,
) -> tuple[float | None, Spectrum]:
    """Frequency of the highest spectral peak inside *band_hz*.

    The window is Hann-tapered and zero-padded by *zero_pad_factor* (to the
    next power of 2); the peak is refined by parabolic interpolation of the
    log-power.

    Returns
    -------
    tuple[float or None, Spectrum]
        ``(rate_hz, spectrum)``; ``rate_hz`` is ``None`` if the band holds no
        frequency of the spectrum.
    """
    n = len(x)
    n_fft = int(2 ** np.ceil(np.log2(max(n, 2) * zero_pad_factor)))
    spectrum = periodogram(x, f_s_hz, n_fft=n_fft)
    in_band = (spectrum.f_hz >= band_hz[0]) & (spectrum.f_hz <= band_hz[1])
    if not np.any(in_band):
        return None, spectrum
    band_indices = np.flatnonzero(in_band)
    k = int(band_indices[np.argmax(spectrum.psd[band_indices])])
    if 0 < k < spectrum.psd.size - 1:
        log_power = np.log(spectrum.psd[k - 1: k + 2] + 1e-300)
        k_refined = k + _parabolic_offset(log_power[0], log_power[1], log_power[2])
    else:
        k_refined = float(k)
    bin_width_hz = spectrum.f_hz[1] - spectrum.f_hz[0]
    return float(k_refined * bin_width_hz), spectrum


def normalised_acf(x: np.ndarray) -> np.ndarray:
    """Unbiased normalised autocorrelation for lags ``0 … N−1``.

    ``R[τ] = (N / (N − τ)) · Σ x[n] x[n+τ] / Σ x[n]²`` (mean removed).  The
    unbiased correction compensates the triangular taper of the biased
    estimate, which would favour short lags; values may slightly exceed 1 at
    large lags (few terms), so callers should restrict the lag range.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    acf = correlate(x, x, mode="full", method="fft")[n - 1:]
    if acf[0] <= 0:
        return np.zeros(n)
    lags = np.arange(n)
    return acf / acf[0] * (n / (n - lags))


def acf_rate(
    x: np.ndarray,
    f_s_hz: float,
    band_hz: tuple[float, float],
    max_lag_fraction: float = 0.5,
) -> tuple[float | None, float]:
    """Breathing rate from the autocorrelation peak.

    The searched lags are ``[1/high, 1/low]``, capped at
    ``max_lag_fraction · N`` so that the unbiased estimate keeps enough terms.

    Returns
    -------
    tuple[float or None, float]
        ``(rate_hz, acf_peak)``; ``acf_peak`` ∈ [−1, 1] measures the
        periodicity (1 = perfectly periodic).
    """
    low_hz, high_hz = band_hz
    n = len(x)
    if n < 4 or low_hz <= 0 or high_hz <= low_hz:
        return None, 0.0
    acf = normalised_acf(x)
    lag_min = max(1, int(np.ceil(f_s_hz / high_hz)))
    lag_max = min(int(np.floor(f_s_hz / low_hz)), int(max_lag_fraction * n))
    if lag_max <= lag_min:
        return None, 0.0
    searched = acf[lag_min: lag_max + 1]
    # Take the *first* local maximum reaching 80 % of the best one: the
    # maxima at 2, 3 ... periods are nearly as high and would halve the rate.
    peaks, _ = find_peaks(searched)
    if peaks.size:
        best = searched[peaks].max()
        strong_enough = np.flatnonzero(searched[peaks] >= best - 0.2 * abs(best))
        k = int(peaks[strong_enough[0]])
    else:
        k = int(np.argmax(searched))
    if 0 < k < searched.size - 1:
        k_refined = k + _parabolic_offset(searched[k - 1], searched[k], searched[k + 1])
    else:
        k_refined = float(k)
    lag = lag_min + k_refined
    return float(f_s_hz / lag), float(np.clip(searched[k], -1.0, 1.0))


def peak_count_rate(
    x: np.ndarray,
    f_s_hz: float,
    band_hz: tuple[float, float],
    prominence_std: float = 0.5,
) -> float | None:
    """Breathing rate from the mean interval between successive peaks.

    Peaks closer than ``1 / high`` are merged and must have a prominence of
    at least ``prominence_std`` standard deviations of the signal.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4 or x.std() == 0:
        return None
    peaks, _ = find_peaks(
        x, distance=max(1, int(f_s_hz / band_hz[1])), prominence=prominence_std * x.std(),
    )
    if peaks.size < 2:
        return None
    rate_hz = float(f_s_hz / np.mean(np.diff(peaks)))
    return rate_hz if band_hz[0] <= rate_hz <= band_hz[1] else None


def zero_crossing_rate(
    x: np.ndarray,
    f_s_hz: float,
    band_hz: tuple[float, float],
) -> float | None:
    """Breathing rate from interpolated zero-crossing instants.

    Two crossings per period: ``rate = (N_crossings − 1) / (2 · (t_last − t_first))``.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    negative = np.signbit(x)
    crossings = np.flatnonzero(negative[1:] != negative[:-1])
    if crossings.size < 3:
        return None
    # Linear interpolation of each crossing instant between two samples.
    t_s = (crossings + x[crossings] / (x[crossings] - x[crossings + 1])) / f_s_hz
    rate_hz = float((t_s.size - 1) / (2.0 * (t_s[-1] - t_s[0])))
    return rate_hz if band_hz[0] <= rate_hz <= band_hz[1] else None


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
