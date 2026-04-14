"""TX waveform generation for the micro-Doppler radar pipeline."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_tx_buffer(
    mode: str,
    buffer_size: int,
    f_s: float,
    f_offset: float = 0.0,
) -> np.ndarray:
    """Generate a complex baseband TX waveform of length *buffer_size*.

    Parameters
    ----------
    mode : str
        Emission mode. ``"cw"`` produces a constant-envelope carrier
        (all samples equal to 1+0j).  ``"cw_offset"`` produces a complex
        sinusoid at *f_offset* Hz, which shifts the useful signal away
        from the DC bin in the received spectrum.
    buffer_size : int
        Number of IQ samples in the transmit buffer (e.g. 16384).
    f_s : float
        ADC / DAC sampling rate (Hz), e.g. 2.0e6.
    f_offset : float, optional
        Frequency offset in baseband (Hz).  Only used when
        *mode* = ``"cw_offset"``.  Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex64 array of shape ``(buffer_size,)`` with values in [-1, 1]
        (unit amplitude).

    Raises
    ------
    ValueError
        If *mode* is not one of ``{"cw", "cw_offset"}``.
    ValueError
        If *f_offset* violates the Nyquist criterion (|f_offset| >= f_s / 2).

    Notes
    -----
    In a CW micro-Doppler radar the transmit signal is a pure tone at carrier
    frequency f_c.  The PlutoSDR up-converts the baseband buffer to RF, so:

    * **CW mode** — the baseband signal is a DC value (constant 1+0j).
      The RF output is a pure tone at exactly f_c.
    * **CW-offset mode** — the baseband signal is exp(j·2π·f_offset·t),
      producing an RF tone at f_c + f_offset.  This moves the reflected
      signal away from the large DC leakage caused by limited TX/RX
      isolation, reducing the dynamic-range burden on clutter suppression.
    """
    if mode not in ("cw", "cw_offset"):
        raise ValueError(
            f"Unknown emission mode '{mode}'. Expected 'cw' or 'cw_offset'."
        )

    if mode == "cw":
        logger.info("Génération du buffer TX — mode CW (module constant)")
        tx_buffer = np.ones(buffer_size, dtype=np.complex64)
        return tx_buffer

    if abs(f_offset) >= f_s / 2:
        raise ValueError(
            f"f_offset={f_offset} Hz dépasse la fréquence de Nyquist "
            f"(f_s/2 = {f_s / 2} Hz). Réduire f_offset ou augmenter f_s."
        )

    logger.info(
        "Génération du buffer TX — mode CW-offset (f_offset = %.0f Hz)",
        f_offset,
    )
    t = np.arange(buffer_size, dtype=np.float64) / f_s
    tx_buffer = np.exp(1j * 2 * np.pi * f_offset * t).astype(np.complex64)
    return tx_buffer
