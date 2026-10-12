"""Doppler filtering of the slow-time IQ (M1) and audit of the anti-aliasing of a decimation chain.

Place in the chain: between the decimation to the intermediate Doppler rate
(100 Hz) and the slow time (20 Hz), **before** the phase demodulation
(MOTION_ROBUSTNESS_PLAN.md §3.2).  The received IQ is a linear sum::

    x(t) = C(t) + a·exp(jφ_target(t)) + Σ_k b_k(t)·exp(jφ_k(t))

but the circle fit and the arctangent are not linear: a disturber left in
the IQ (gestures at 3–16 Hz, walking at 16–25 Hz at 2.45 GHz) is mixed into
the breathing band by the demodulation.  A complex low-pass filter on the IQ
(same real taps on I and Q) removes everything above the breathing band
first; it also serves as the anti-aliasing filter of the last decimation.

The filter is a linear-phase FIR (Kaiser window): causal, with a constant
delay of ``(N − 1)/2`` samples, i.e. 1.8 s at 100 Hz for the default
template (pass band to 1 Hz, −6 dB at 1.5 Hz, ≥ 60 dB from 2 Hz).

Reference for the multistage decimation and its aliasing bands:
R. E. Crochiere and L. R. Rabiner, "Interpolation and decimation of digital
signals — A tutorial review", Proc. IEEE 69(3), 1981.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, freqz, group_delay, kaiserord

from iot_radar.dsp.decimation import Decimator


def design_doppler_lowpass(
    f_s_hz: float,
    cutoff_hz: float = 1.5,
    transition_hz: float = 1.0,
    attenuation_db: float = 60.0,
) -> np.ndarray:
    """Taps of the linear-phase Doppler low-pass filter (Kaiser window).

    Parameters
    ----------
    f_s_hz : float
        Sampling rate of the filtered IQ (Hz), e.g. 100 Hz.
    cutoff_hz : float, optional
        −6 dB frequency (Hz).
    transition_hz : float, optional
        Width of the transition band (Hz), centred on *cutoff_hz*: the pass
        band ends at ``cutoff − transition/2`` and the stop band starts at
        ``cutoff + transition/2``.
    attenuation_db : float, optional
        Minimum stop-band attenuation (dB); also bounds the pass-band ripple.

    Returns
    -------
    numpy.ndarray
        Symmetric taps (odd length: integer delay of ``(N − 1)/2`` samples).

    Notes
    -----
    Kaiser's formula ``N ≈ (A − 7.95) / (2.285 · 2π · Δf / f_s)`` gives 365
    taps for the default template at 100 Hz (Crochiere & Rabiner, 1981, on
    FIR design for decimation).
    """
    n_taps, beta = kaiserord(attenuation_db, transition_hz / (0.5 * f_s_hz))
    if n_taps % 2 == 0:
        n_taps += 1
    return firwin(n_taps, cutoff_hz, window=("kaiser", beta), fs=f_s_hz)


def doppler_lowpass_decimator(
    f_s_hz: float,
    decimation_factor: int,
    cutoff_hz: float = 1.5,
    transition_hz: float = 1.0,
    attenuation_db: float = 60.0,
) -> tuple[Decimator, float]:
    """Streaming decimator whose only stage is the Doppler low-pass filter.

    Returns
    -------
    (Decimator, float)
        The decimator (state kept between blocks, ``reset()`` on a
        discontinuity) and the delay it introduces (s).
    """
    taps = design_doppler_lowpass(f_s_hz, cutoff_hz, transition_hz, attenuation_db)
    decimator = Decimator(f_s_hz, decimation_factor, max_frequency_hz=cutoff_hz,
                          stages=[(taps, np.array([1.0]), decimation_factor)])
    return decimator, (len(taps) - 1) / 2.0 / f_s_hz


@dataclass
class AliasStage:
    """Aliasing into the band of interest at one decimation stage.

    Attributes
    ----------
    index : int
        Rank of the stage in the chain (0 = first).
    f_in_hz, f_out_hz : float
        Input and output rates of the stage (Hz).
    decimation_factor : int
        Factor ``q`` of the stage.
    worst_attenuation_db : float
        Smallest attenuation of the stage filter over the bands
        ``m·f_out ± band`` (m ≥ 1) that fold into ``[−band, +band]`` (dB).
    worst_frequency_hz : float
        Input frequency of that worst case (Hz).
    """

    index: int
    f_in_hz: float
    f_out_hz: float
    decimation_factor: int
    worst_attenuation_db: float
    worst_frequency_hz: float


def _attenuation_db(b: np.ndarray, a: np.ndarray, frequencies_hz: np.ndarray, f_s_hz: float) -> np.ndarray:
    _, response = freqz(b, a, worN=np.abs(frequencies_hz), fs=f_s_hz)
    return -20.0 * np.log10(np.abs(response) + 1e-300)


def alias_audit(stages: list[tuple[np.ndarray, np.ndarray, int, float]], band_hz: float = 1.0,
                points_per_band: int = 41) -> list[AliasStage]:
    """Attenuation of every stage in the frequency bands that fold into the band of interest.

    For a stage of input rate ``F`` and factor ``q``, the frequencies
    ``m·F/q ± band`` (``m = 1 … ⌊(F/2 + band)/(F/q)⌋``) fold into
    ``[−band, +band]`` at its output (Crochiere & Rabiner, 1981, §II).

    Parameters
    ----------
    stages : list
        ``(b, a, q, f_in_hz)`` of every stage, in order (``Decimator.stages``).
    band_hz : float, optional
        Half-width ``f_b`` of the band of interest (Hz).
    points_per_band : int, optional
        Frequencies evaluated in each folding band.
    """
    report = []
    for index, (b, a, q, f_in_hz) in enumerate(stages):
        f_out_hz = f_in_hz / q
        worst, worst_f = np.inf, float("nan")
        for m in range(1, int(np.floor((f_in_hz / 2 + band_hz) / f_out_hz)) + 1):
            f = np.linspace(m * f_out_hz - band_hz, min(m * f_out_hz + band_hz, f_in_hz / 2), points_per_band)
            attenuation = _attenuation_db(b, a, f, f_in_hz)
            if attenuation.min() < worst:
                worst, worst_f = float(attenuation.min()), float(f[np.argmin(attenuation)])
        report.append(AliasStage(index, f_in_hz, f_out_hz, int(q), worst, worst_f))
    return report


def composite_alias_attenuation(
    stages: list[tuple[np.ndarray, np.ndarray, int, float]],
    band_hz: float = 1.0,
    max_frequency_hz: float | None = None,
    n_points: int = 200_001,
) -> tuple[float, float]:
    """Worst attenuation of an input tone that ends up in ``[−band, +band]`` after the whole chain.

    Each input frequency is followed through the stages: attenuated by the
    filter of a stage, then folded by its decimation.  Among the tones
    outside the band that end inside it, the least attenuated one is the
    worst case, whatever its folding path.

    Returns
    -------
    (float, float)
        Its attenuation (dB) and its input frequency (Hz).
    """
    f_in_hz = stages[0][3]
    max_frequency_hz = max_frequency_hz or f_in_hz / 2
    f_hz = np.linspace(-max_frequency_hz, max_frequency_hz, n_points)
    attenuation_db = np.zeros_like(f_hz)
    folded_hz = f_hz.copy()
    for b, a, q, stage_in_hz in stages:
        attenuation_db += _attenuation_db(b, a, folded_hz, stage_in_hz)
        f_out_hz = stage_in_hz / q
        folded_hz = (folded_hz + f_out_hz / 2) % f_out_hz - f_out_hz / 2
    aliased = (np.abs(folded_hz) <= band_hz) & (np.abs(f_hz) > band_hz)
    if not aliased.any():
        return float("inf"), float("nan")
    worst = int(np.argmin(np.where(aliased, attenuation_db, np.inf)))
    return float(attenuation_db[worst]), float(f_hz[worst])


def dc_group_delay_s(stages: list[tuple[np.ndarray, np.ndarray, int, float]]) -> float:
    """Delay of a decimation chain for the slow components (group delay near 0 Hz, s)."""
    delay_s = 0.0
    for b, a, _, f_in_hz in stages:
        _, delay_samples = group_delay((b, a), w=[1e-6 * f_in_hz], fs=f_in_hz)
        delay_s += float(delay_samples[0]) / f_in_hz
    return delay_s
