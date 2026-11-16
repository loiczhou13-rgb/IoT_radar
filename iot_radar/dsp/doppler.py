"""Doppler low-pass filter of the slow-time IQ (M1) and aliasing audit of a decimation chain.

Place in the chain: between the decimation to 100 Hz and the slow time
(20 Hz), before the demodulation.  The circle fit and the arctangent are not
linear: a disturber left in the IQ (gestures 3–16 Hz, walking 16–25 Hz) would
be mixed into the breathing band.  A complex low-pass on the IQ removes it
first and serves as the anti-aliasing filter of the last ×5 decimation.

Template (Kaiser FIR, 365 taps, delay 1.82 s): flat to 1 Hz (breathing
harmonics), −6 dB at 1.5 Hz, ≥ 60 dB from 2 Hz.  100 Hz keeps the walking
Doppler below Nyquist.  Worst aliasing into ±1 Hz: 84 dB (88 dB for the
former Chebyshev chain, which let gestures through up to 8 Hz).

Reference: R. E. Crochiere and L. R. Rabiner, "Interpolation and decimation of
digital signals — A tutorial review", Proc. IEEE 69(3), 1981.
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
    """Symmetric taps (odd length) of the Kaiser low-pass: −6 dB at *cutoff_hz*,
    transition of *transition_hz* centred on it, *attenuation_db* in the stop band.

    Length from Kaiser's formula (``scipy.signal.kaiserord``): 365 taps at 100 Hz.
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
    """Streaming decimator made of the Doppler low-pass alone, and its delay (s)."""
    taps = design_doppler_lowpass(f_s_hz, cutoff_hz, transition_hz, attenuation_db)
    decimator = Decimator(f_s_hz, decimation_factor, max_frequency_hz=cutoff_hz,
                          stages=[(taps, np.array([1.0]), decimation_factor)])
    return decimator, (len(taps) - 1) / 2.0 / f_s_hz


@dataclass
class AliasStage:
    """Aliasing into ``[−band, +band]`` at one stage: rates, factor, and the
    smallest attenuation (dB) over the folding bands with its input frequency (Hz)."""

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
    """Attenuation of each stage ``(b, a, q, f_in_hz)`` over the bands ``m·f_out ± band``
    that fold into ``[−band, +band]`` (Crochiere & Rabiner, 1981)."""
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
    """Worst ``(attenuation_db, input_frequency_hz)`` of an out-of-band tone that
    ends in ``[−band, +band]`` after the whole chain (filtered, then folded, stage by stage)."""
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
    """Delay (s) of a decimation chain near 0 Hz."""
    delay_s = 0.0
    for b, a, _, f_in_hz in stages:
        _, delay_samples = group_delay((b, a), w=[1e-6 * f_in_hz], fs=f_in_hz)
        delay_s += float(delay_samples[0]) / f_in_hz
    return delay_s
