"""Digital frequency shift (NCO): brings the echo from ``+tx_offset_hz`` to 0 Hz.

Place in the chain: before the decimation, whose low-pass would otherwise
remove the echo.  The receiver DC moves to ``−tx_offset_hz`` and is filtered out.
"""

from __future__ import annotations

import numpy as np


class Mixer:
    """Shift a complex stream by ``−shift_hz``, block by block.

    The phase is carried between blocks (kept in [0, 2π)), so the output does
    not depend on how the stream is cut.
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
