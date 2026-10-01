"""Digital frequency shift (numerically controlled oscillator, NCO).

Place in the chain: right after the acquisition, before the decimation.  In
``cw_offset`` mode the echo is received at ``+tx_offset_hz``; multiplying the
samples by ``exp(−j·2π·tx_offset_hz·t)`` brings it back to 0 Hz **before**
the low-pass filter of the decimation (which would otherwise remove it).  The
receiver DC offset moves to ``−tx_offset_hz`` at the same time and is
rejected by the decimation filter.
"""

from __future__ import annotations

import numpy as np


class Mixer:
    """Shift the spectrum of a complex stream by ``−shift_hz``, block by block.

    Parameters
    ----------
    f_s_hz : float
        Sampling rate of the stream (Hz).
    shift_hz : float
        Frequency brought to 0 Hz (Hz); ``0`` leaves the samples unchanged.

    Notes
    -----
    The oscillator phase is carried from one block to the next (and kept in
    [0, 2π) to avoid a loss of precision), so the output does not depend on
    how the stream is cut into blocks.
    """

    def __init__(self, f_s_hz: float, shift_hz: float) -> None:
        self.f_s_hz = float(f_s_hz)
        self.shift_hz = float(shift_hz)
        self._phase_step_rad = -2.0 * np.pi * self.shift_hz / self.f_s_hz
        self._phase_rad = 0.0

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """Shifted copy of one block of complex samples (complex128, same length)."""
        x = np.asarray(samples, dtype=np.complex128)
        if self.shift_hz == 0.0:
            return x
        phase_rad = self._phase_rad + self._phase_step_rad * np.arange(x.size)
        shifted = x * np.exp(1j * phase_rad)
        self._phase_rad = float(np.mod(self._phase_rad + self._phase_step_rad * x.size, 2.0 * np.pi))
        return shifted

    def reset(self) -> None:
        """Restart the oscillator at phase 0 (after a discontinuity of the stream)."""
        self._phase_rad = 0.0
