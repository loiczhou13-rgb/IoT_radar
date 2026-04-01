"""
Transmit baseband waveform generation for PlutoSDR micro-Doppler operation.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

EmissionMode = Literal["cw", "cw_offset"]


def generate_tx_waveform(
    mode: EmissionMode,
    n_samples: int,
    f_s: float,
    f_offset_hz: float = 0.0,
    amplitude: float = 2.0**14,
) -> np.ndarray:
    """
    Build a complex baseband buffer for cyclic transmission on PlutoSDR.

    The nominal amplitude follows the common Pluto tutorial scaling (peak
    magnitude ``2**14``) before hardware-specific scaling in the acquisition
    layer.

    Parameters
    ----------
    mode
        ``"cw"``: constant envelope, zero frequency in baseband.
        ``"cw_offset"``: single tone at ``f_offset_hz`` to move TX energy away
        from DC after downconversion (typical offset: a few kHz).
    n_samples
        Number of complex samples in one cyclic buffer (unit: samples).
    f_s
        Sample rate used to define the discrete-time tone (unit: Hz,
        typical value ``2.0e6``).
    f_offset_hz
        Tone frequency for ``cw_offset`` mode (unit: Hz, typical ``10_000``).
    amplitude
        Peak magnitude of the complex envelope (unit: dimensionless DAC scale,
        typical ``16384`` = ``2**14``).

    Returns
    -------
    numpy.ndarray
        Complex64 array of shape ``(n_samples,)`` containing the TX buffer.

    Examples
    --------
    >>> iq = generate_tx_waveform("cw", 1024, 2e6)
    >>> iq.shape
    (1024,)

    Notes
    -----
    Physical rationale: a stable CW (or offset-CW) illuminates the scene so
    that chest motion imposes a slow phase modulation on the echo; the
    micro-Doppler line at the breathing rate appears in the baseband spectrum.
    A small intentional frequency offset can reduce leakage from DC clutter and
    IQ imbalance in the receiver chain.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    n_samples = int(n_samples)
    f_s = float(f_s)

    if mode == "cw":
        x = np.ones(n_samples, dtype=np.complex64) * float(amplitude)
        logger.debug("Generated CW TX buffer: n=%d, |x|=%g", n_samples, amplitude)

    elif mode == "cw_offset":
        t = np.arange(n_samples, dtype=np.float64) / f_s
        phase = 2.0 * np.pi * float(f_offset_hz) * t
        x = (float(amplitude) * np.exp(1j * phase)).astype(np.complex64)
        logger.debug(
            "Generated CW-offset TX buffer: n=%d, f_offset=%g Hz, |x|=%g",
            n_samples,
            f_offset_hz,
            amplitude,
        )

    else:
        raise ValueError(f"Unknown emission mode: {mode!r}")

    return x


def scale_for_pyadi_tx(iq: np.ndarray) -> np.ndarray:
    """
    Scale emission buffers to the range expected by ``pyadi-iio`` Pluto drivers.

    Many Analog Devices examples use ``2**11`` scaling when calling ``tx()``;
    our nominal design peak is ``2**14``, so we apply a fixed ratio.

    Parameters
    ----------
    iq
        Complex waveform from :func:`generate_tx_waveform` (unit: nominal DAC
        counts with peak ``~2**14``).

    Returns
    -------
    numpy.ndarray
        Complex64 array of the same shape, scaled for ``adi.Pluto.tx``.

    Examples
    --------
    >>> x = generate_tx_waveform("cw", 8, 1e6)
    >>> y = scale_for_pyadi_tx(x)
    >>> np.isfinite(y).all()
    True

    Notes
    -----
    Physical note: correct scaling avoids DAC clipping while keeping enough
    SNR on the echo; saturation on RX is checked separately after capture.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    scale = (2.0**11) / (2.0**14)
    return (iq * scale).astype(np.complex64)
