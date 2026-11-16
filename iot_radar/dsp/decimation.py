"""Decimation of complex IQ streams (ADC rate → slow time).

Place in the chain: after the mixer.  The factor is split into primes ≤ 13,
one stage each: Chebyshev I order-8 low-pass (cut-off ``0.8 / q`` of the stage
Nyquist, as ``scipy.signal.decimate``), then one sample out of ``q``.  The
filters are causal and keep their state between blocks, so the block
boundaries are invisible in the output.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import cheby1, lfilter

logger = logging.getLogger(__name__)

_MAX_PRIME = 13
_PRIMES = (2, 3, 5, 7, 11, 13)
_CHEBY_ORDER = 8
_CHEBY_RIPPLE_DB = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _factorise(decimation_factor: int) -> list[int]:
    """Prime factors ≤ 13 of *decimation_factor*, ascending (``ValueError`` otherwise)."""
    if decimation_factor <= 1:
        return []

    factors: list[int] = []
    remaining = decimation_factor

    for p in _PRIMES:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p

    if remaining > 1:
        raise ValueError(
            f"decimation factor {decimation_factor} has a prime factor > {_MAX_PRIME} "
            f"(residue {remaining}). Choose a factor built from small primes "
            f"({', '.join(map(str, _PRIMES))})."
        )

    factors.sort()
    return factors


def _check_shannon(f_s_hz: float, decimation_factor: int, max_frequency_hz: float) -> None:
    """``ValueError`` if the output rate is below 2.5 × *max_frequency_hz* (2 + filter margin)."""
    f_s_out_hz = f_s_hz / decimation_factor
    if decimation_factor > 1 and f_s_out_hz <= 2.5 * max_frequency_hz:
        raise ValueError(
            f"Shannon criterion (with anti-aliasing margin) violated: output rate "
            f"{f_s_out_hz:.1f} Hz <= 2.5 x max_frequency_hz = "
            f"{2.5 * max_frequency_hz:.1f} Hz. Reduce the decimation factor "
            f"(currently {decimation_factor}) or increase the sampling rate."
        )


# ---------------------------------------------------------------------------
# Decimator
# ---------------------------------------------------------------------------

class Decimator:
    """Stateful cascaded decimator of a complex stream (I and Q filtered separately).

    *max_frequency_hz* only serves the Shannon check; *output_dtype*
    ``complex128`` when the cascade is cut in two (the cut then changes
    nothing); *stages* ``[(b, a, q), ...]`` replace the Chebyshev cascade
    (e.g. the M1 FIR).
    """

    def __init__(
        self,
        f_s_hz: float,
        decimation_factor: int,
        max_frequency_hz: float,
        output_dtype: Any = np.complex64,
        stages: list[tuple[np.ndarray, np.ndarray, int]] | None = None,
    ) -> None:
        self.f_s_in_hz = float(f_s_hz)
        self.decimation_factor = int(decimation_factor)
        self.f_s_out_hz = self.f_s_in_hz / self.decimation_factor

        _check_shannon(self.f_s_in_hz, self.decimation_factor, max_frequency_hz)

        self._stages: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._zi_re: list[np.ndarray] = []
        self._zi_im: list[np.ndarray] = []
        self._phase: list[int] = []
        self._output_dtype = output_dtype

        if stages is None:
            stages = [(*cheby1(_CHEBY_ORDER, _CHEBY_RIPPLE_DB, 0.8 / q), q)
                      for q in _factorise(self.decimation_factor)]
        elif int(np.prod([q for _, _, q in stages])) != self.decimation_factor:
            raise ValueError("The product of the stage factors must be the decimation factor.")
        for b, a, q in stages:
            b, a = np.atleast_1d(np.asarray(b, dtype=np.float64)), np.atleast_1d(np.asarray(a, dtype=np.float64))
            n_state = max(len(a), len(b)) - 1
            self._stages.append((b, a, q))
            self._zi_re.append(np.zeros(n_state, dtype=np.float64))
            self._zi_im.append(np.zeros(n_state, dtype=np.float64))
            self._phase.append(0)

        logger.info(
            "Decimator — factor %d in %d stage(s), %.0f Hz -> %.1f Hz",
            self.decimation_factor, len(self._stages), self.f_s_in_hz, self.f_s_out_hz,
        )

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        """Filter and down-sample one block; on average ``len(iq) / factor`` samples out."""
        if self.decimation_factor == 1:
            return iq.astype(self._output_dtype, copy=False)

        re = iq.real.astype(np.float64, copy=False)
        im = iq.imag.astype(np.float64, copy=False)

        for i, (b, a, q) in enumerate(self._stages):
            if re.size == 0:
                continue

            re, self._zi_re[i] = lfilter(b, a, re, zi=self._zi_re[i])
            im, self._zi_im[i] = lfilter(b, a, im, zi=self._zi_im[i])

            # First sample to keep: the kept samples stay q apart across blocks.
            offset = self._phase[i]
            n = re.size

            if offset >= n:
                self._phase[i] = offset - n
                re = re[:0]
                im = im[:0]
                continue

            keep = np.arange(offset, n, q)
            self._phase[i] = int((keep[-1] + q) - n)
            re = re[keep]
            im = im[keep]

        return (re + 1j * im).astype(self._output_dtype)

    @property
    def stages(self) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
        """``(b, a, q, f_in_hz)`` of every stage (for the aliasing audit)."""
        stages = []
        f_in_hz = self.f_s_in_hz
        for b, a, q in self._stages:
            stages.append((b, a, q, f_in_hz))
            f_in_hz /= q
        return stages

    def reset(self) -> None:
        """Clear the filter states and sub-sampling phases."""
        for i in range(len(self._stages)):
            self._zi_re[i][:] = 0.0
            self._zi_im[i][:] = 0.0
            self._phase[i] = 0
