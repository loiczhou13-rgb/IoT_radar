"""PlutoSDR (ADALM-PLUTO, AD9363) transceiver: TX waveform, configuration, RX checks.

Place in the chain: first step.  The Pluto is used as a monostatic CW radar:
its transmitter replays a baseband waveform in **cyclic** mode (a pure carrier,
or a tone at ``offset_hz``), and its receiver, tuned to the same local
oscillator, delivers the echoes already down-converted to baseband.

This module only talks to the hardware; :class:`iot_radar.acquisition.sources.PlutoSource`
turns the received buffers into blocks for the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ADC_FULL_SCALE: int = 2048
"""Full scale of the 12-bit ADC as returned by ``pyadi-iio``: samples in [-2048, 2047]."""

_ADC_SATURATION_RATIO: float = 0.80

RX_STATUS_REGISTER: int = 0x80000088
"""Status register of the RX DMA (``cf-ad9361-lpc`` device, AXI address space)."""

RX_OVERFLOW_BIT: int = 0b0100
"""Bit set by the FPGA when RX samples were dropped (host too slow)."""

DAC_FULL_SCALE: int = 2**14
"""Amplitude that maps unit-amplitude IQ to the DAC full scale with ``pyadi-iio``.

The ``adi.Pluto.tx()`` API casts ``complex64`` samples directly to ``int16``
without any scaling.  The AD9363 DAC is 12-bit but ``pyadi-iio`` aligns its
samples to the upper bits of the ``int16`` word, so unit-amplitude IQ has to
be multiplied by ``2**14`` to reach the DAC full scale.  Without this scaling
the carrier sits at ~1 LSB (≈ −84 dBFS) and is invisible on a spectrum
analyser.
"""


# ---------------------------------------------------------------------------
# Transmitted waveform
# ---------------------------------------------------------------------------

def cw_tx_buffer(
    waveform: str,
    buffer_size: int,
    f_s_hz: float,
    offset_hz: float = 0.0,
) -> np.ndarray:
    """Cyclic baseband TX buffer of a CW radar.

    Parameters
    ----------
    waveform : str
        ``"cw"`` (constant sample: pure carrier) or ``"cw_offset"`` (complex
        tone at *offset_hz*, which moves the useful signal away from the
        receiver DC offset).
    buffer_size : int
        Number of IQ samples of the buffer.
    f_s_hz : float
        DAC sampling rate (Hz).
    offset_hz : float, optional
        Baseband frequency of the tone (Hz), ``"cw_offset"`` only.  Must be a
        whole number of periods per buffer (see :func:`snap_tx_offset_hz`).

    Returns
    -------
    numpy.ndarray
        Complex64 array of shape ``(buffer_size,)`` scaled to the PlutoSDR
        DAC full scale (``±2**14``, see :data:`DAC_FULL_SCALE`).

    Raises
    ------
    ValueError
        If *waveform* is unknown, if *offset_hz* violates the Nyquist
        criterion (``|offset_hz| >= f_s_hz / 2``), or if the buffer would not
        hold a whole number of periods of the tone.

    Notes
    -----
    * ``"cw"`` — samples ``2**14 + 0j``: RF tone at exactly the LO frequency.
    * ``"cw_offset"`` — samples ``2**14 · exp(j·2π·offset_hz·t)``: RF tone at
      ``LO + offset_hz``.
    * The Pluto replays the buffer back-to-back (cyclic mode).  With a
      non-integer number of periods per buffer, the phase would jump at
      every repetition and the emitted spectrum would become a set of lines
      at multiples of ``f_s_hz / buffer_size`` (bug B1: 500 Hz requested,
      dominant line at 488.28 Hz).
    """
    if waveform not in ("cw", "cw_offset"):
        raise ValueError(
            f"Unknown TX waveform '{waveform}'. Expected 'cw' or 'cw_offset'."
        )

    if waveform == "cw":
        logger.info("TX buffer — CW (constant envelope, scale=%d)", DAC_FULL_SCALE)
        return np.full(buffer_size, DAC_FULL_SCALE + 0j, dtype=np.complex64)

    if abs(offset_hz) >= f_s_hz / 2:
        raise ValueError(
            f"offset_hz={offset_hz} Hz exceeds the Nyquist frequency "
            f"(f_s/2 = {f_s_hz / 2} Hz). Reduce the offset or increase the sampling rate."
        )

    periods_per_buffer = offset_hz * buffer_size / f_s_hz
    if abs(periods_per_buffer - round(periods_per_buffer)) > 1e-6:
        raise ValueError(
            f"offset_hz={offset_hz} Hz gives {periods_per_buffer:.4f} periods per "
            f"{buffer_size}-sample buffer: the cyclic buffer would be "
            f"discontinuous. Use snap_tx_offset_hz() (multiple of "
            f"{f_s_hz / buffer_size:.4f} Hz)."
        )

    logger.info(
        "TX buffer — CW offset (offset=%.0f Hz, scale=%d)",
        offset_hz,
        DAC_FULL_SCALE,
    )
    t_s = np.arange(buffer_size, dtype=np.float64) / f_s_hz
    waveform_samples = DAC_FULL_SCALE * np.exp(1j * 2 * np.pi * offset_hz * t_s)
    return waveform_samples.astype(np.complex64)


def snap_tx_offset_hz(requested_offset_hz: float, f_s_hz: float, buffer_size: int) -> float:
    """Nearest offset giving a whole number of periods per TX buffer.

    The cyclic TX buffer is continuous only if it holds an integer number of
    periods of the offset tone, i.e. if the offset is a multiple of
    ``f_s_hz / buffer_size`` (122.07 Hz for 16384 samples at 2 MS/s).

    Parameters
    ----------
    requested_offset_hz : float
        Requested baseband offset (Hz).
    f_s_hz : float
        DAC sampling rate (Hz).
    buffer_size : int
        Number of samples of the cyclic TX buffer.

    Returns
    -------
    float
        Effective offset (Hz): ``round(requested / grid) * grid`` with
        ``grid = f_s_hz / buffer_size``.

    Raises
    ------
    ValueError
        If the snapped offset is 0 (requested offset below ``grid / 2``).
    """
    grid_hz = f_s_hz / buffer_size
    n_periods = round(requested_offset_hz / grid_hz)
    if n_periods == 0:
        raise ValueError(
            f"offset {requested_offset_hz} Hz is too small: the smallest possible "
            f"offset is {grid_hz:.4f} Hz (f_s / buffer_size)."
        )
    return n_periods * grid_hz


def effective_tx_offset_hz(cfg: dict[str, Any]) -> float:
    """TX offset actually transmitted (Hz), 0 if the waveform is not ``"cw_offset"``.

    The configured ``tx.offset_hz`` is snapped with :func:`snap_tx_offset_hz`;
    the snapped value is the one transmitted, so it is also the one the
    receiver must use (bug B1).
    """
    tx_cfg = cfg.get("tx", {})
    if tx_cfg.get("waveform") != "cw_offset":
        return 0.0
    requested_hz = float(tx_cfg.get("offset_hz", 0.0))
    sdr = cfg["sdr"]
    effective_hz = snap_tx_offset_hz(requested_hz, float(sdr["sample_rate_hz"]), int(sdr["buffer_size"]))
    if effective_hz != requested_hz:
        logger.info(
            "TX offset snapped from %.3f Hz to %.5f Hz (whole number of periods "
            "per %d-sample TX buffer)",
            requested_hz,
            effective_hz,
            int(sdr["buffer_size"]),
        )
    return effective_hz


# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------

def open_pluto(
    uri: str,
    f_c_hz: float,
    f_s_hz: float,
    rx_gain_db: float,
    tx_gain_db: float,
    buffer_size: int,
    tx_buffer: np.ndarray,
):
    """Connect to the PlutoSDR, configure it and start the cyclic transmission.

    Parameters
    ----------
    uri : str
        PlutoSDR address, e.g. ``"ip:192.168.2.1"`` or ``"usb:"``.
    f_c_hz, f_s_hz : float
        Carrier (TX and RX local oscillator) frequency and ADC sampling rate (Hz).
        The analog RF bandwidth is set to *f_s_hz*.
    rx_gain_db, tx_gain_db : float
        Manual receiver gain (dB) and transmitter attenuation (dB, ≤ 0).
    buffer_size : int
        Samples per RX buffer.
    tx_buffer : numpy.ndarray
        Complex64 baseband TX waveform (cyclic), **already scaled to ±2**14**
        (see :func:`cw_tx_buffer`).

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
            "pyadi-iio is not installed: pip install pyadi-iio"
        ) from exc

    try:
        sdr = adi.Pluto(uri)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach the PlutoSDR at '{uri}'. Check the IP address "
            f"(e.g. ip:192.168.2.1) or the USB link (usb:)."
        ) from exc

    sdr.sample_rate = int(f_s_hz)
    sdr.rx_lo = int(f_c_hz)
    sdr.tx_lo = int(f_c_hz)
    sdr.rx_rf_bandwidth = int(f_s_hz)
    sdr.tx_rf_bandwidth = int(f_s_hz)
    sdr.rx_buffer_size = buffer_size
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = rx_gain_db
    sdr.tx_hardwaregain_chan0 = tx_gain_db

    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_buffer)
    return sdr


# ---------------------------------------------------------------------------
# Reception checks
# ---------------------------------------------------------------------------

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
    """Warn if an RX buffer approaches the ADC full scale (80 %).

    Saturation clips the echoes; reduce ``sdr.rx_gain_db`` or
    ``sdr.tx_gain_db`` if it happens.
    """
    peak = np.max(np.abs(frame))
    threshold = _ADC_SATURATION_RATIO * ADC_FULL_SCALE
    if peak > threshold:
        logger.warning(
            "Probable ADC saturation — buffer %d: max(|IQ|) = %.0f "
            "(threshold %.0f, full scale %d)",
            frame_index,
            peak,
            threshold,
            ADC_FULL_SCALE,
        )
