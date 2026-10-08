"""Contamination of the slow-time IQ by people moving around the radar, and sample weights (M3).

Place in the chain: at the intermediate Doppler rate (100 Hz), **before** the
Doppler filter M1, which would remove the evidence (MOTION_ROBUSTNESS_PLAN.md
§3.4).  For each block of ``block_s`` seconds, the energy of the IQ above the
breathing band (``out_of_band_start_hz`` up to half the rate: gestures,
walking) is compared with its energy in the breathing band (DC — the static
part — excluded), both normalised by their bandwidths::

    κ = 10·log10( (E_out / B_out) / (E_in / B_in) )      [dB]

so that white noise gives 0 dB whatever the rates and bands.  A breathing
target alone gives κ ≪ 0 dB, a disturber κ ≫ 0 dB.  The weight of the block
is a smooth function of κ — no binary pause::

    w = 1 / (1 + exp((κ − κ₀) / s))

The weights feed the weighted, gap-tolerant estimation of the breathing
rate (:func:`iot_radar.dsp.estimation.weighted_lomb_scargle`; J. T.
VanderPlas, "Understanding the Lomb-Scargle periodogram", ApJS 236(1), 2018).

Each block also gets an *activity*: its out-of-band density relative to the
floor of that density (low quantile over the last minute, in the spirit of
the minimum-statistics noise estimate of R. Martin, IEEE T-SAP 9(5), 2001).
It tells that someone moves around the radar, whatever the strength of the
target, and serves to attribute an in-band anomaly to a passer-by rather
than to the target (MOTION_ROBUSTNESS_PLAN.md §3.4).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def contamination_weight(index_db: float | np.ndarray, midpoint_db: float = 6.0,
                         slope_db: float = 2.0) -> float | np.ndarray:
    """Weight in [0, 1] of a block of contamination index *index_db*.

    Logistic: 0.95 at 0 dB (white noise), 0.5 at *midpoint_db*, 0.05 at
    ``midpoint_db + 6 dB`` with the default slope of 2 dB.
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
        Out-of-band density relative to its recent floor (dB, ≥ 0 up to noise).
    """

    sample_start: int
    index_db: float
    weight: float
    activity_db: float = 0.0


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
    floor_window_s : float, optional
        Length of the past over which the floor of the out-of-band density
        is taken (s).
    floor_quantile : float, optional
        Quantile of the out-of-band densities taken as their floor.
    """

    def __init__(
        self,
        f_s_hz: float,
        block_s: float = 1.0,
        out_of_band_start_hz: float = 2.0,
        in_band_hz: tuple[float, float] = (0.1, 1.0),
        midpoint_db: float = 6.0,
        slope_db: float = 2.0,
        order: int = 4,
        history_s: float = 120.0,
        floor_window_s: float = 60.0,
        floor_quantile: float = 0.1,
    ) -> None:
        self.f_s_hz = float(f_s_hz)
        self.block_length = max(1, int(round(block_s * self.f_s_hz)))
        self.midpoint_db, self.slope_db = float(midpoint_db), float(slope_db)
        self._highpass = butter(order, out_of_band_start_hz, btype="highpass", fs=self.f_s_hz, output="sos")
        self._bandpass = butter(order, in_band_hz, btype="bandpass", fs=self.f_s_hz, output="sos")
        # Two-sided bandwidths of a complex signal.
        self._out_bandwidth_hz = 2.0 * (self.f_s_hz / 2.0 - out_of_band_start_hz)
        self._in_bandwidth_hz = 2.0 * (in_band_hz[1] - in_band_hz[0])
        self.blocks: deque[ContaminationBlock] = deque(maxlen=max(1, int(history_s / block_s)))
        self._out_densities: deque[float] = deque(maxlen=max(1, int(round(floor_window_s / block_s))))
        self.floor_quantile = float(floor_quantile)
        self.reset()

    def reset(self) -> None:
        """Forget the past (after a discontinuity of the stream)."""
        out_shape, in_shape = (self._highpass.shape[0], 2), (self._bandpass.shape[0], 2)
        self._state = {"out_re": np.zeros(out_shape), "out_im": np.zeros(out_shape),
                       "in_re": np.zeros(in_shape), "in_im": np.zeros(in_shape)}
        self._out_energy = 0.0
        self._in_energy = 0.0
        self._in_block = 0          # samples already in the current block
        self._block_start = 0       # index of the first sample of the current block
        self._started = False
        self._block_saturated = False
        self.blocks.clear()
        self._out_densities.clear()

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
            for name, sos, value in (("out_re", self._highpass, x[0].real), ("out_im", self._highpass, x[0].imag),
                                     ("in_re", self._bandpass, x[0].real), ("in_im", self._bandpass, x[0].imag)):
                self._state[name] = sosfilt_zi(sos) * value
            self._started = True
        out_re, self._state["out_re"] = sosfilt(self._highpass, x.real, zi=self._state["out_re"])
        out_im, self._state["out_im"] = sosfilt(self._highpass, x.imag, zi=self._state["out_im"])
        in_re, self._state["in_re"] = sosfilt(self._bandpass, x.real, zi=self._state["in_re"])
        in_im, self._state["in_im"] = sosfilt(self._bandpass, x.imag, zi=self._state["in_im"])
        out_power = out_re ** 2 + out_im ** 2
        in_power = in_re ** 2 + in_im ** 2
        position = 0
        while position < x.size:
            take = min(self.block_length - self._in_block, x.size - position)
            self._out_energy += float(out_power[position:position + take].sum())
            self._in_energy += float(in_power[position:position + take].sum())
            self._in_block += take
            position += take
            if saturated and position == x.size:
                self._block_saturated = True
            if self._in_block == self.block_length:
                self._close_block()

    def _close_block(self) -> None:
        out_density = self._out_energy / self._out_bandwidth_hz
        in_density = self._in_energy / self._in_bandwidth_hz
        index_db = float(10.0 * np.log10(max(out_density, 1e-300) / max(in_density, 1e-300)))
        weight = 0.0 if self._block_saturated else float(contamination_weight(index_db, self.midpoint_db,
                                                                                   self.slope_db))
        self._out_densities.append(out_density)
        floor = float(np.quantile(np.array(self._out_densities), self.floor_quantile))
        activity_db = float(10.0 * np.log10(max(out_density, 1e-300) / max(floor, 1e-300)))
        self.blocks.append(ContaminationBlock(self._block_start, index_db, weight, activity_db))
        self._block_start += self.block_length
        self._out_energy = self._in_energy = 0.0
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
