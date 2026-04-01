"""
Sample-rate decimation with anti-aliasing (scipy.signal.decimate).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def decimate_iq(
    x: np.ndarray,
    f_s: float,
    factor: int,
    f_max_utile_hz: float,
    enabled: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Decimate complex IQ by an integer factor with Chebyshev low-pass filtering.

    Enforces Shannon: ``f_s / D > 2 * f_max_utile`` when ``enabled``.

    Parameters
    ----------
    x
        Input complex baseband IQ, shape ``(N,)`` (unitless complex samples).
    f_s
        Input sample rate (unit: Hz, typical ``2.0e6``).
    factor
        Integer decimation factor ``D`` (typical ``1000``).
    f_max_utile_hz
        Highest frequency that must remain unaliased after decimation
        (unit: Hz, typical ``10`` for respiration harmonics margin).
    enabled
        If ``False``, returns ``x`` unchanged and ``f_s`` unchanged.

    Returns
    -------
    tuple[numpy.ndarray, float]
        ``(x_dec, f_s_new)`` where ``x_dec`` has shape roughly ``(N // D,)`` and
        ``f_s_new = f_s / D``.

    Examples
    --------
    >>> x = np.exp(1j * np.linspace(0, 10, 5000)).astype(np.complex64)
    >>> y, fs2 = decimate_iq(x, 1e6, 10, 40.0)
    >>> fs2
    100000.0

    Notes
    -----
    Physical rationale: raw ADC rates are MHz-scale while chest motion lives
    below ~10 Hz; decimation lowers the STFT cost and maps Doppler bins to
    meaningful Hertz spacing, provided the low-pass rejects aliases of strong
    clutter and wideband noise.
    """
    x = np.asarray(x, dtype=np.complex64)
    f_s = float(f_s)
    if not enabled or int(factor) == 1:
        logger.info("Decimation disabled; f_s stays %.6e Hz", f_s)
        return x, f_s

    D = int(factor)
    if D < 1:
        raise ValueError("Decimation factor D must be >= 1")

    f_new = f_s / D
    nyq_ok = f_new > 2.0 * float(f_max_utile_hz)
    if not nyq_ok:
        raise ValueError(
            "Shannon criterion violated after decimation: require f_s / D > 2 * f_max_utile "
            f"but got f_s/D = {f_new:.6g} Hz with f_max_utile = {f_max_utile_hz} Hz "
            f"(2*f_max_utile = {2.0 * f_max_utile_hz} Hz). "
            "Reduce D or increase f_s, or lower f_max_utile."
        )

    logger.info(
        "Decimating with D=%d: f_s %.6e Hz -> %.6e Hz; usable Doppler span ±%.6e Hz",
        D,
        f_s,
        f_new,
        f_new / 2.0,
    )

    # scipy decimate real/imag independently preserves envelope better than
    # decimating magnitude/phase separately for narrowband IQ.
    xr = signal.decimate(x.real, D, ftype="iir", zero_phase=True)
    xi = signal.decimate(x.imag, D, ftype="iir", zero_phase=True)
    x_dec = (xr + 1j * xi).astype(np.complex64)
    return x_dec, f_new
