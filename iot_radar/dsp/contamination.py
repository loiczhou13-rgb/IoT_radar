"""Contamination of the IQ by people moving around the radar, and sample weights (M3).

Place in the chain: at the Doppler rate (100 Hz), before M1, which would
remove the evidence.  Per block of ``block_s``:

* contamination index (densities per Hz, so white noise gives 0 dB)::

      κ = 10·log10( (E_out / B_out) / (E_in / B_in) )      [dB]

  out = above ``out_of_band_start_hz`` (gestures, walking), in = breathing
  band without DC.  Simulation: breathing alone −30 to −20 dB, walker at
  3.5 m ≈ −5 dB, gestures at 1.5 m ≈ 0 dB.
* weight ``w = 1 / (1 + exp((κ − κ₀) / s))``, used by the weighted
  Lomb-Scargle estimation (no binary pause);
* activity: density in the human Doppler band (3–28 Hz) over a noise-only
  band (30–38 Hz).  ≈ 0 ± 3 dB when nobody moves, whatever the target;
  walker at 3–6 m +8 to +22 dB.  Tells that someone moves around, to
  attribute an in-band anomaly to a passer-by rather than to the target.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def contamination_weight(index_db: float | np.ndarray, midpoint_db: float = 9.0,
                         slope_db: float = 2.0) -> float | np.ndarray:
    """Logistic weight of a block: 0.99 at 0 dB, 0.5 at *midpoint_db*, 0.05 at +6 dB.

    9 dB from the sweep of ``scripts/evaluate_motion_robustness.py``: Pd 85 %
    (72 % at 6 dB), 8 % inconclusive with a target (20 % at 6 dB), no false alarm.
    """
    return 1.0 / (1.0 + np.exp((np.asarray(index_db, dtype=np.float64) - midpoint_db) / slope_db))


@dataclass
class ContaminationBlock:
    """One block: first sample (Doppler rate, since the reset), κ (dB), weight,
    activity (dB) and ADC saturation (weight 0)."""

    sample_start: int
    index_db: float
    weight: float
    activity_db: float = 0.0
    saturated: bool = False


class ContaminationMeter:
    """Streaming contamination index of the IQ at the Doppler rate *f_s_hz*, block by block (M3).

    The bands are those of the module docstring; *history_s* is the history
    kept for :meth:`block_at`.
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
        """Add a chunk of samples; *saturated* (ADC full scale) gives its block weight 0."""
        x = np.asarray(x, dtype=np.complex128)
        if x.size == 0:
            return
        if not self._started:
            # Start in steady state, or the step of the static echo rings for tens of seconds.
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
        """Block of *sample_index* (Doppler rate), or the latest complete one; ``None`` before the first."""
        if not self.blocks:
            return None
        k = (sample_index - self.blocks[0].sample_start) // self.block_length
        return self.blocks[min(max(k, 0), len(self.blocks) - 1)]

    @property
    def latest(self) -> ContaminationBlock | None:
        """The latest complete block, or ``None``."""
        return self.blocks[-1] if self.blocks else None
