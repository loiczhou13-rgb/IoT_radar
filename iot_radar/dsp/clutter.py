"""Static-clutter suppression for the micro-Doppler spectrogram chain.

Place in the chain: after the decimation, before the STFT.  Static echoes
(walls, TX->RX leakage) and the receiver DC offset produce a huge line at
0 Hz that would dominate the spectrogram; :class:`ClutterFilter` removes the
component around 0 Hz with a high-pass filter.

.. warning::
   A high-pass filter also removes the mean of the useful term, which
   distorts the phase of the signal: it is acceptable for a magnitude
   product such as a spectrogram, never before a phase demodulation.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

logger = logging.getLogger(__name__)

_VALID_MODES = ("mean", "iir", "mti", "butterworth")


class ClutterFilter:
    """Stateful clutter-suppression filter for streaming pipelines.

    Parameters
    ----------
    mode : str
        Clutter-removal strategy:

        * ``"mean"`` — subtraction of a running mean (exponential moving
          average, EMA), computed sample by sample;
        * ``"iir"`` — the same EMA high-pass, vectorised with ``lfilter``;
        * ``"mti"`` — first difference ``y[n] = x[n] - x[n-1]`` (deprecated:
          it destroys the breathing frequencies);
        * ``"butterworth"`` — Butterworth high-pass filter.
    f_s_hz : float
        Sampling rate of the input signal (Hz).
    alpha : float, optional
        EMA memory factor of the ``"iir"`` and ``"mean"`` modes (close to 1:
        long memory, low cut-off ``≈ (1 - alpha) · f_s / 2π``).
    butterworth_order : int, optional
        Order of the ``"butterworth"`` high-pass.
    butterworth_cutoff_hz : float, optional
        Cut-off frequency of the ``"butterworth"`` high-pass (Hz).

    Notes
    -----
    The internal state (running mean, previous sample, filter state) is kept
    between successive calls, so that a stream can be processed block by
    block.  In ``"butterworth"`` and ``"mti"`` modes the output does not
    depend on how the signal is cut into blocks.  In ``"iir"`` and
    ``"mean"`` modes the running mean starts at the mean of the **first**
    block, so only the initial transient depends on the first block length.
    """

    def __init__(
        self,
        mode: str,
        f_s_hz: float,
        alpha: float = 0.9999,
        butterworth_order: int = 2,
        butterworth_cutoff_hz: float = 0.05,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown clutter mode '{mode}'. "
                f"Use {', '.join(repr(m) for m in _VALID_MODES)}."
            )
        self._mode = mode
        self._f_s_hz = f_s_hz
        self._alpha = alpha
        self._mu: complex | None = None
        self._prev: complex | None = None

        if mode in ("iir", "mean"):
            cutoff_hz = (1.0 - alpha) * f_s_hz / (2.0 * math.pi)
            logger.info(
                "ClutterFilter — mode='%s', alpha=%.4f, cut-off ≈ %.3f Hz (f_s=%.1f Hz)",
                mode,
                alpha,
                cutoff_hz,
                f_s_hz,
            )

        if mode == "mti":
            logger.warning(
                "MTI mode is unsuited to breathing detection: the filter "
                "y[n] = x[n] - x[n-1] attenuates 0.3 Hz by ~60 dB at f_s = %.0f Hz "
                "and DESTROYS the breathing signal. Prefer 'iir' or 'butterworth'.",
                f_s_hz,
            )

        if mode == "butterworth":
            self._sos = butter(
                butterworth_order,
                butterworth_cutoff_hz,
                btype="high",
                fs=f_s_hz,
                output="sos",
            )
            self._zi_real = None
            self._zi_imag = None
            logger.info(
                "ClutterFilter — mode='butterworth', order=%d, cut-off=%.3f Hz (f_s=%.1f Hz)",
                butterworth_order,
                butterworth_cutoff_hz,
                f_s_hz,
            )

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        """Filter one block of IQ samples.

        Parameters
        ----------
        iq : numpy.ndarray
            Complex IQ samples (1-D).

        Returns
        -------
        numpy.ndarray
            Clutter-suppressed IQ, same length (and, except in ``"mti"``
            mode, same dtype) as *iq*.
        """
        if iq.ndim != 1:
            raise ValueError(
                f"iq must be 1-D, got ndim={iq.ndim} shape={iq.shape}."
            )
        if not np.iscomplexobj(iq):
            logger.warning(
                "iq is not complex (dtype=%s) — cast to complex128.",
                iq.dtype,
            )
            iq = iq.astype(np.complex128)

        if self._mode == "mean":
            return self._apply_mean(iq)
        if self._mode == "iir":
            return self._apply_iir(iq)
        if self._mode == "butterworth":
            return self._apply_butterworth(iq)
        return self._apply_mti(iq)

    def _apply_mean(self, iq: np.ndarray) -> np.ndarray:
        """Subtract a running EMA mean, sample by sample."""
        out = np.empty_like(iq)
        mu = self._mu if self._mu is not None else complex(np.mean(iq))
        alpha = self._alpha
        for n in range(len(iq)):
            mu = alpha * mu + (1.0 - alpha) * iq[n]
            out[n] = iq[n] - mu
        self._mu = mu
        return out

    def _apply_iir(self, iq: np.ndarray) -> np.ndarray:
        """Vectorised EMA high-pass (``lfilter``) with state carry-over."""
        alpha = self._alpha
        b = np.array([1.0 - alpha])
        a = np.array([1.0, -alpha])

        if not hasattr(self, "_zi_iir"):
            mu0 = complex(np.mean(iq))
            self._zi_iir = np.array([mu0 * alpha])

        mu_filtered, self._zi_iir = lfilter(b, a, iq, zi=self._zi_iir)
        self._mu = complex(mu_filtered[-1])
        return (iq - mu_filtered).astype(iq.dtype)

    def _apply_mti(self, iq: np.ndarray) -> np.ndarray:
        """Single-delay MTI (first difference) with state carry-over."""
        prev = self._prev if self._prev is not None else complex(iq[0])
        extended = np.concatenate(([prev], iq))
        self._prev = complex(iq[-1])
        return np.diff(extended)

    def _apply_butterworth(self, iq: np.ndarray) -> np.ndarray:
        """Butterworth high-pass with state carry-over (I and Q separately)."""
        if self._zi_real is None:
            # Start in steady state on the first sample (no initial step).
            zi = sosfilt_zi(self._sos)
            self._zi_real = zi * iq.real[0]
            self._zi_imag = zi * iq.imag[0]

        y_real, self._zi_real = sosfilt(
            self._sos, iq.real.astype(np.float64), zi=self._zi_real,
        )
        y_imag, self._zi_imag = sosfilt(
            self._sos, iq.imag.astype(np.float64), zi=self._zi_imag,
        )
        return (y_real + 1j * y_imag).astype(iq.dtype)
