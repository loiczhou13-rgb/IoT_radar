"""Contamination of the slow-time IQ by people moving around the radar, and sample weights (M3).

Place in the chain: at the intermediate Doppler rate (100 Hz), **before** the
Doppler filter M1, which would remove the evidence.  For each block of ``block_s`` seconds, the energy of the IQ above the
breathing band (``out_of_band_start_hz`` up to half the rate: gestures,
walking) is compared with its energy in the breathing band (DC — the static
part — excluded), both normalised by their bandwidths::

    κ = 10·log10( (E_out / B_out) / (E_in / B_in) )      [dB]

so that white noise gives 0 dB whatever the rates and bands.  Measured in
simulation: a breathing target alone gives −30 to −20 dB, a walker at 3.5 m
raises κ to about −5 dB, gestures at 1.5 m to about 0 dB.  Because κ compares
densities, an out-of-band line is diluted over the ~96 Hz out-of-band
bandwidth: the weight only drops for a disturber clearly stronger than the
breathing echo (about 10 times in amplitude for w = 0.5); a weaker one is
already removed linearly by M1.  The weight of the block
is a smooth function of κ — no binary pause::

    w = 1 / (1 + exp((κ − κ₀) / s))

The weights feed the weighted, gap-tolerant estimation of the breathing
rate (:func:`iot_radar.dsp.estimation.weighted_lomb_scargle`).

Each block also gets an *activity*: the density in the band of the human
Doppler (3–28 Hz: gestures up to 16 Hz, walking up to 25 Hz at 2.45 GHz;
from 3 Hz so that the sidebands of a deep breath stay out) over the density
in a band above it (30–38 Hz), which only holds the receiver noise and is
still inside the pass band of the decimation::

    activity = 10·log10( (E_act / B_act) / (E_ref / B_ref) )      [dB]

About 0 dB when nobody moves (±3 dB from block to block), whatever the
strength of the target and whatever the past: a disturber present from the
start is still seen, which a floor taken on the past of the out-of-band
density would miss.  Measured in simulation: a walker at 3–6 m gives +8 to
+22 dB, a deep breath (3 rad of excursion) less than 5 dB.  The first block
after a restart holds the transient of the filters and counts as 0 dB.  The
activity tells that someone moves around the radar, and serves to attribute
an in-band anomaly to a passer-by rather than to the target.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def contamination_weight(index_db: float | np.ndarray, midpoint_db: float = 9.0,
                         slope_db: float = 2.0) -> float | np.ndarray:
    """Weight in [0, 1] of a block of contamination index *index_db*.

    Logistic: 0.99 at 0 dB (white noise), 0.5 at *midpoint_db*, 0.05 at
    ``midpoint_db + 6 dB`` with the default slope of 2 dB.  The default
    midpoint of 9 dB comes from a sweep of the evaluation
    (``scripts/evaluate_motion_robustness.py --midpoints-db 0 3 6 9 12``,
    M1 + M3, walkers and gestures with and without a target, 30 s
    measurements): 9 dB gives Pd 85 % (6 dB: 72 %), 8 % of inconclusive
    measurements with a target (6 dB: 20 %), no false alarm and no pause;
    12 dB changes nothing more.  With 6 dB, gestures that M1 already removes
    made a quarter of the measurements inconclusive.
    """
    return 1.0 / (1.0 + np.exp((np.asarray(index_db, dtype=np.float64) - midpoint_db) / slope_db))


@dataclass
class ContaminationBlock:
    """Contamination of one block of samples.

    Attributes
    ----------
    sample_start : int
        Index of its first sample at the Doppler rate (since the last reset).
    index_db : float
        Contamination index κ (dB).
    weight : float
        Weight of its samples, in [0, 1].
    activity_db : float
        Density in the band of the human Doppler over the noise density (dB, ≈ 0 when nobody moves).
    saturated : bool
        Some of its samples come from ADC samples at full scale (weight 0).
    """

    sample_start: int
    index_db: float
    weight: float
    activity_db: float = 0.0
    saturated: bool = False


class ContaminationMeter:
    """Streaming contamination index, block by block (M3).

    Parameters
    ----------
    f_s_hz : float
        Rate of the IQ (Hz), the intermediate Doppler rate.
    block_s : float, optional
        Length of a block (s).
    out_of_band_start_hz : float, optional
        Lower edge of the out-of-band energy (Hz); it ends at ``f_s_hz / 2``.
    in_band_hz : tuple[float, float], optional
        Band of the in-band energy (Hz); its lower edge removes the static part.
    midpoint_db, slope_db : float, optional
        Parameters of :func:`contamination_weight`.
    order : int, optional
        Butterworth order of the two streaming filters.
    history_s : float, optional
        Length of the block history kept for :meth:`block_at` (s).
    activity_band_hz, reference_band_hz : tuple[float, float], optional
        Band of the human Doppler and noise-only band above it (Hz), both
        inside the pass band of the decimation to ``f_s_hz``.
    """

    def __init__(
        self,
        f_s_hz: float,
        block_s: float = 1.0,
        out_of_band_start_hz: float = 2.0,
        in_band_hz: tuple[float, float] = (0.1, 1.0),
        midpoint_db: float = 9.0,
        slope_db: float = 2.0,
        order: int = 4,
        history_s: float = 120.0,
        activity_band_hz: tuple[float, float] = (3.0, 28.0),
        reference_band_hz: tuple[float, float] = (30.0, 38.0),
    ) -> None:
        self.f_s_hz = float(f_s_hz)
        self.block_length = max(1, int(round(block_s * self.f_s_hz)))
        self.midpoint_db, self.slope_db = float(midpoint_db), float(slope_db)
        self._filters = {
            "out": butter(order, out_of_band_start_hz, btype="highpass", fs=self.f_s_hz, output="sos"),
            "in": butter(order, in_band_hz, btype="bandpass", fs=self.f_s_hz, output="sos"),
            "act": butter(order, activity_band_hz, btype="bandpass", fs=self.f_s_hz, output="sos"),
            "ref": butter(order, reference_band_hz, btype="bandpass", fs=self.f_s_hz, output="sos"),
        }
        # Two-sided bandwidths of a complex signal.
        self._bandwidths_hz = {
            "out": 2.0 * (self.f_s_hz / 2.0 - out_of_band_start_hz),
            "in": 2.0 * (in_band_hz[1] - in_band_hz[0]),
            "act": 2.0 * (activity_band_hz[1] - activity_band_hz[0]),
            "ref": 2.0 * (reference_band_hz[1] - reference_band_hz[0]),
        }
        self.blocks: deque[ContaminationBlock] = deque(maxlen=max(1, int(history_s / block_s)))
        self.reset()

    def reset(self) -> None:
        """Forget the past (after a discontinuity of the stream)."""
        self._state = {name: (np.zeros((sos.shape[0], 2)), np.zeros((sos.shape[0], 2)))
                       for name, sos in self._filters.items()}           # real and imaginary parts
        self._energies = dict.fromkeys(self._filters, 0.0)
        self._in_block = 0          # samples already in the current block
        self._block_start = 0       # index of the first sample of the current block
        self._started = False
        self._block_saturated = False
        self.blocks.clear()

    def process(self, x: np.ndarray, saturated: bool = False) -> None:
        """Take a chunk of complex samples at ``f_s_hz`` into account.

        *saturated*: the chunk comes from ADC samples at full scale; the
        block that receives its end gets weight 0.
        """
        x = np.asarray(x, dtype=np.complex128)
        if x.size == 0:
            return
        if not self._started:
            # Filters started in steady state on the first sample: the step of
            # the static echo would otherwise ring in the band-pass for tens
            # of seconds and pass for in-band energy.
            self._state = {name: (sosfilt_zi(sos) * x[0].real, sosfilt_zi(sos) * x[0].imag)
                           for name, sos in self._filters.items()}
            self._started = True
        powers = {}
        for name, sos in self._filters.items():
            real, state_re = sosfilt(sos, x.real, zi=self._state[name][0])
            imag, state_im = sosfilt(sos, x.imag, zi=self._state[name][1])
            self._state[name] = (state_re, state_im)
            powers[name] = real ** 2 + imag ** 2
        position = 0
        while position < x.size:
            take = min(self.block_length - self._in_block, x.size - position)
            for name, power in powers.items():
                self._energies[name] += float(power[position:position + take].sum())
            self._in_block += take
            position += take
            if saturated and position == x.size:
                self._block_saturated = True
            if self._in_block == self.block_length:
                self._close_block()

    def _close_block(self) -> None:
        density = {name: max(self._energies[name] / self._bandwidths_hz[name], 1e-300) for name in self._filters}
        index_db = float(10.0 * np.log10(density["out"] / density["in"]))
        # The first block after a (re)start holds the transient of the filters.
        activity_db = float(10.0 * np.log10(density["act"] / density["ref"])) if self.blocks else 0.0
        weight = 0.0 if self._block_saturated else float(contamination_weight(index_db, self.midpoint_db,
                                                                                   self.slope_db))
        self.blocks.append(ContaminationBlock(self._block_start, index_db, weight, activity_db,
                                              self._block_saturated))
        self._block_start += self.block_length
        self._energies = dict.fromkeys(self._filters, 0.0)
        self._in_block = 0
        self._block_saturated = False

    def block_at(self, sample_index: int) -> ContaminationBlock | None:
        """Block of the sample *sample_index* (Doppler rate, since the last reset).

        Its block once the block is complete; until then, the latest complete
        block; ``None`` before the first block.  The blocks are contiguous, so
        the block of a sample is found by division.
        """
        if not self.blocks:
            return None
        k = (sample_index - self.blocks[0].sample_start) // self.block_length
        return self.blocks[min(max(k, 0), len(self.blocks) - 1)]

    @property
    def latest(self) -> ContaminationBlock | None:
        """The latest complete block, or ``None``."""
        return self.blocks[-1] if self.blocks else None
