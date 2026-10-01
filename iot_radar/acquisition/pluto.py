"""PlutoSDR (ADALM-PLUTO) transceiver: TX waveform and streaming reception."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ADC_FULL_SCALE: int = 2048
_ADC_SATURATION_RATIO: float = 0.80

RX_STATUS_REGISTER: int = 0x80000088
"""Status register of the RX DMA (``cf-ad9361-lpc`` device, AXI address space)."""

RX_OVERFLOW_BIT: int = 0b0100
"""Bit set by the FPGA when RX samples were dropped (host too slow)."""

_DAC_FULL_SCALE: int = 2**14
"""PlutoSDR DAC convention used by ``pyadi-iio``.

The ``adi.Pluto.tx()`` API casts ``complex64`` samples directly to
``int16`` without applying any scaling.  The AD9363 DAC is 12-bit but
``pyadi-iio`` aligns its samples to the upper bits of the ``int16`` word,
so unit-amplitude IQ has to be multiplied by ``2**14`` to reach DAC
full-scale.  Without this scaling the carrier sits at ~1 LSB
(≈ −84 dBFS) and is invisible on a spectrum analyser.
"""


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
        Emission mode. ``"cw"`` produces a constant-envelope carrier;
        ``"cw_offset"`` produces a complex sinusoid at *f_offset* Hz,
        which shifts the useful signal away from the DC bin.
    buffer_size : int
        Number of IQ samples in the transmit buffer.
    f_s : float
        ADC / DAC sampling rate (Hz).
    f_offset : float, optional
        Frequency offset in baseband (Hz).  Only used when
        *mode* = ``"cw_offset"``.  Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex64 array of shape ``(buffer_size,)`` with samples scaled
        to the PlutoSDR DAC full-scale (``±2**14``).

    Raises
    ------
    ValueError
        If *mode* is not one of ``{"cw", "cw_offset"}``, if *f_offset*
        violates the Nyquist criterion (``|f_offset| >= f_s / 2``), or if
        the buffer would not hold a whole number of periods of the offset
        tone (use :func:`snap_tx_offset`).

    Notes
    -----
    * **CW mode** — baseband samples are a constant ``2**14 + 0j``;
      the RF output is a pure tone at exactly ``f_c``.
    * **CW-offset mode** — baseband samples are
      ``2**14 · exp(j·2π·f_offset·t)``, producing an RF tone at
      ``f_c + f_offset`` and avoiding the DC clutter.
    * The Pluto replays the buffer back-to-back (cyclic mode).  With a
      non-integer number of periods per buffer, the phase would jump at
      every repetition and the emitted spectrum would become a set of lines
      at multiples of ``f_s / buffer_size`` (bug B1: 500 Hz requested,
      dominant line at 488.28 Hz).

    The ``2**14`` scaling is **mandatory**: without it the DAC effectively
    transmits a zero-amplitude signal (see :data:`_DAC_FULL_SCALE`).
    """
    if mode not in ("cw", "cw_offset"):
        raise ValueError(
            f"Unknown emission mode '{mode}'. Expected 'cw' or 'cw_offset'."
        )

    if mode == "cw":
        logger.info(
            "Génération du buffer TX — mode CW (module constant, scale=%d)",
            _DAC_FULL_SCALE,
        )
        return np.full(buffer_size, _DAC_FULL_SCALE + 0j, dtype=np.complex64)

    if abs(f_offset) >= f_s / 2:
        raise ValueError(
            f"f_offset={f_offset} Hz dépasse la fréquence de Nyquist "
            f"(f_s/2 = {f_s / 2} Hz). Réduire f_offset ou augmenter f_s."
        )

    periods_per_buffer = f_offset * buffer_size / f_s
    if abs(periods_per_buffer - round(periods_per_buffer)) > 1e-6:
        raise ValueError(
            f"f_offset={f_offset} Hz donne {periods_per_buffer:.4f} périodes par "
            f"buffer de {buffer_size} échantillons : le buffer cyclique serait "
            f"discontinu. Utiliser snap_tx_offset() (multiple de "
            f"{f_s / buffer_size:.4f} Hz)."
        )

    logger.info(
        "Génération du buffer TX — mode CW-offset (f_offset=%.0f Hz, scale=%d)",
        f_offset,
        _DAC_FULL_SCALE,
    )
    t = np.arange(buffer_size, dtype=np.float64) / f_s
    waveform = _DAC_FULL_SCALE * np.exp(1j * 2 * np.pi * f_offset * t)
    return waveform.astype(np.complex64)


def snap_tx_offset(f_offset: float, f_s: float, buffer_size: int) -> float:
    """Nearest offset giving a whole number of periods per TX buffer.

    The cyclic TX buffer is continuous only if it holds an integer number of
    periods of the offset tone, i.e. if ``f_offset`` is a multiple of
    ``f_s / buffer_size`` (122.07 Hz for 16384 samples at 2 MS/s).

    Parameters
    ----------
    f_offset : float
        Requested baseband offset (Hz).
    f_s : float
        DAC sampling rate (Hz).
    buffer_size : int
        Number of samples of the cyclic TX buffer.

    Returns
    -------
    float
        Effective offset (Hz): ``round(f_offset / grid) * grid`` with
        ``grid = f_s / buffer_size``.

    Raises
    ------
    ValueError
        If the snapped offset is 0 (requested offset below ``grid / 2``).
    """
    grid_hz = f_s / buffer_size
    n_periods = round(f_offset / grid_hz)
    if n_periods == 0:
        raise ValueError(
            f"f_offset={f_offset} Hz est trop faible : le plus petit décalage "
            f"possible est {grid_hz:.4f} Hz (f_s / buffer_size)."
        )
    return n_periods * grid_hz


def resolve_f_offset(cfg: dict[str, Any]) -> float:
    """Return the effective baseband offset (Hz), 0 if not in cw_offset mode.

    The configured ``emission.f_offset`` is snapped with
    :func:`snap_tx_offset`; the snapped value is the one transmitted, so it
    is also the one used by the receiver (bug B1).
    """
    emi = cfg.get("emission", {})
    if emi.get("mode") != "cw_offset":
        return 0.0
    requested = float(emi.get("f_offset", 0.0))
    sdr = cfg["sdr"]
    effective = snap_tx_offset(requested, float(sdr["f_s"]), int(sdr["buffer_size"]))
    if effective != requested:
        logger.info(
            "f_offset recalé de %.3f Hz à %.5f Hz (nombre entier de périodes "
            "par buffer TX de %d échantillons)",
            requested,
            effective,
            int(sdr["buffer_size"]),
        )
    return effective


def open_pluto(
    uri: str,
    f_c: float,
    f_s: float,
    rx_gain: float,
    tx_gain: float,
    buffer_size: int,
    tx_buffer: np.ndarray,
):
    """Connect to the PlutoSDR, configure it and start the cyclic transmission.

    Parameters
    ----------
    uri : str
        PlutoSDR address, e.g. ``"ip:192.168.2.1"`` or ``"usb:"``.
    f_c, f_s : float
        Carrier frequency (Hz) and ADC sampling rate (Hz).
    rx_gain, tx_gain : float
        Receiver gain (dB) and transmitter attenuation (dB, negative).
    buffer_size : int
        Samples per RX buffer.
    tx_buffer : numpy.ndarray
        Complex64 baseband TX waveform (cyclic).  **Already scaled to
        ±2**14** — see :func:`generate_tx_buffer`.

    Returns
    -------
    adi.Pluto
        The configured device; the TX path replays *tx_buffer* indefinitely
        (**cyclic** mode) while the caller receives with ``sdr.rx()``.

    Raises
    ------
    RuntimeError
        If ``pyadi-iio`` is missing or the PlutoSDR cannot be reached at *uri*.
    """
    try:
        import adi
    except ImportError as exc:
        raise RuntimeError(
            "pyadi-iio n'est pas installé. Exécuter : pip install pyadi-iio"
        ) from exc

    try:
        sdr = adi.Pluto(uri)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de se connecter au PlutoSDR à '{uri}'. "
            f"Vérifier l'adresse IP (ex. ip:192.168.2.1) ou la connexion USB (usb:)."
        ) from exc

    sdr.sample_rate = int(f_s)
    sdr.rx_lo = int(f_c)
    sdr.tx_lo = int(f_c)
    sdr.rx_rf_bandwidth = int(f_s)
    sdr.tx_rf_bandwidth = int(f_s)
    sdr.rx_buffer_size = buffer_size
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = rx_gain
    sdr.tx_hardwaregain_chan0 = tx_gain

    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_buffer)
    return sdr


def clear_rx_overflow(sdr) -> None:
    """Clear the sticky bits of the RX status register.

    Raises whatever ``reg_write`` raises when the register is not accessible.
    """
    sdr._rxadc.reg_write(RX_STATUS_REGISTER, 0x6)


def read_and_clear_rx_overflow(sdr) -> bool:
    """``True`` if the FPGA dropped RX samples since the previous call.

    The bit is cleared after reading (write 1 to clear).  Raises whatever
    ``reg_read`` raises when the register is not accessible.
    """
    status = sdr._rxadc.reg_read(RX_STATUS_REGISTER)
    if status & RX_OVERFLOW_BIT:
        sdr._rxadc.reg_write(RX_STATUS_REGISTER, status)
        return True
    return False


def check_saturation(frame: np.ndarray, frame_index: int) -> None:
    """Warn if the IQ frame approaches ADC saturation."""
    peak = np.max(np.abs(frame))
    threshold = _ADC_SATURATION_RATIO * _ADC_FULL_SCALE
    if peak > threshold:
        logger.warning(
            "Saturation ADC probable — trame %d : max(|IQ|) = %.0f "
            "(seuil = %.0f, pleine échelle = %d)",
            frame_index,
            peak,
            threshold,
            _ADC_FULL_SCALE,
        )
