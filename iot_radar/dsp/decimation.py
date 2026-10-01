"""Sample-rate reduction (decimation) of complex IQ streams.

Place in the chain: right after the acquisition and the NCO mixer
(:mod:`iot_radar.dsp.mixer`).  The PlutoSDR delivers
samples at millions of samples per second, while breathing lives below 1 Hz;
:class:`Decimator` low-pass filters the stream and keeps one sample out of
``decimation_factor``.

:class:`Decimator` is stateful and causal: instantiate it once per stream and
call it block by block.  The filter states and the sub-sampling phase are kept
between calls, so block boundaries are invisible in the output — a complete
recording can also be decimated offline by passing it as a single block (the
result is identical).

The decimation factor is decomposed into prime factors ≤ 13 and one Chebyshev
type-I order-8 anti-aliasing filter (cut-off ``0.8 / q`` of the stage Nyquist
frequency, as in ``scipy.signal.decimate``) is applied per stage.

Note
----
Zero-phase filtering (``filtfilt``) needs the complete signal, so it cannot be
used on a stream.  ``lfilter`` (forward only) introduces a constant group
delay, harmless for signals well below the cut-off (~0.4 x output rate).
"""

from __future__ import annotations

import logging

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
    """Decompose *decimation_factor* into prime factors ≤ 13 (ascending).

    Each factor becomes one decimation stage with its own Chebyshev
    anti-aliasing filter, maximising the stop-band rejection per stage.

    Raises
    ------
    ValueError
        If *decimation_factor* has a prime factor > 13.
    """
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
    """Raise ``ValueError`` if the output rate cannot represent *max_frequency_hz*.

    The guard factor 2.5 (instead of the theoretical 2.0) accounts for the
    transition band of the Chebyshev anti-aliasing filter.
    """
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
    """Stateful cascaded decimator for complex IQ streams.

    Parameters
    ----------
    f_s_hz : float
        Input sampling rate (Hz).
    decimation_factor : int
        Total decimation factor; must factor into primes ≤ 13.  The output
        rate is ``f_s_hz / decimation_factor``.
    max_frequency_hz : float
        Highest frequency of interest (Hz), only used for the Shannon check.

    Notes
    -----
    Each stage applies a Chebyshev type-I order-8 low-pass filter with
    ``Wn = 0.8 / q`` and then keeps one sample out of *q*.  I and Q are
    filtered separately (real filters), each with its own state.
    """

    def __init__(self, f_s_hz: float, decimation_factor: int, max_frequency_hz: float) -> None:
        self.f_s_in_hz = float(f_s_hz)
        self.decimation_factor = int(decimation_factor)
        self.f_s_out_hz = self.f_s_in_hz / self.decimation_factor

        _check_shannon(self.f_s_in_hz, self.decimation_factor, max_frequency_hz)

        self._stages: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._zi_re: list[np.ndarray] = []
        self._zi_im: list[np.ndarray] = []
        self._phase: list[int] = []

        for q in _factorise(self.decimation_factor):
            b, a = cheby1(_CHEBY_ORDER, _CHEBY_RIPPLE_DB, 0.8 / q)
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
        """Low-pass filter and down-sample one block, keeping the filter state.

        Parameters
        ----------
        iq : numpy.ndarray
            Complex IQ samples (1-D) at ``f_s_in_hz``.

        Returns
        -------
        numpy.ndarray
            ``complex64`` samples at ``f_s_out_hz``.  For a steady stream of
            equal blocks the average output length is
            ``len(iq) / decimation_factor``; one call may differ by one
            sample, depending on the sub-sampling phase.
        """
        if self.decimation_factor == 1:
            return iq.astype(np.complex64, copy=False)

        re = iq.real.astype(np.float64, copy=False)
        im = iq.imag.astype(np.float64, copy=False)

        for i, (b, a, q) in enumerate(self._stages):
            if re.size == 0:
                continue

            re, self._zi_re[i] = lfilter(b, a, re, zi=self._zi_re[i])
            im, self._zi_im[i] = lfilter(b, a, im, zi=self._zi_im[i])

            # Index of the first sample to keep in this block, so that the
            # kept samples stay exactly q apart across block boundaries.
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

        return (re + 1j * im).astype(np.complex64)

    def reset(self) -> None:
        """Clear the filter states and sub-sampling phases (start of a new stream)."""
        for i in range(len(self._stages)):
            self._zi_re[i][:] = 0.0
            self._zi_im[i][:] = 0.0
            self._phase[i] = 0
