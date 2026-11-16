"""PlutoSDR (AD9363) used as a monostatic CW radar: TX waveform, configuration, RX checks.

The transmitter replays a baseband buffer in cyclic mode (carrier, or tone at ``offset_hz``); the receiver, on the same LO, delivers the echoes in baseband. :class:`iot_radar.acquisition.sources.PlutoSource` turns the buffers into blocks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ADC_FULL_SCALE: int = 2048
"""Full scale of the 12-bit ADC with ``pyadi-iio``: samples in [-2048, 2047]."""

_ADC_SATURATION_RATIO: float = 0.80

RX_STATUS_REGISTER: int = 0x80000088
"""Status register of the RX DMA (``cf-ad9361-lpc``)."""

RX_OVERFLOW_BIT: int = 0b0100
"""Set by the FPGA when RX samples were dropped (host too slow)."""

DAC_HALF_SCALE: int = 2**14
"""Scale of unit-amplitude TX samples (half the int16 range, −6 dBFS): ``pyadi-iio`` casts to int16 without scaling and the FPGA feeds the upper 12 bits to the DAC (unscaled, all TX samples sit within +/- 1, below one DAC step)."""


# ---------------------------------------------------------------------------
# Transmitted waveform
# ---------------------------------------------------------------------------

def cw_tx_buffer(
    waveform: str,
    buffer_size: int,
    f_s_hz: float,
    offset_hz: float = 0.0,
) -> np.ndarray:
    """Cyclic TX buffer (complex64, scaled to ±2**14): ``"cw"`` (constant sample) or
    ``"cw_offset"`` (tone at *offset_hz*, away from the receiver DC).

    The tone must hold a whole number of periods per buffer
    (:func:`snap_tx_offset_hz`): otherwise the phase jumps at every
    repetition and the spectrum becomes lines every ``f_s / buffer_size``
    (bug B1: 500 Hz requested, line at 488.28 Hz).  ``ValueError`` on an
    unknown waveform, an offset of 0.4·f_s or more (4/5 of Nyquist) or a non-integer period count.
    """
    if waveform not in ("cw", "cw_offset"):
        raise ValueError(
            f"Unknown TX waveform '{waveform}'. Expected 'cw' or 'cw_offset'."
        )

    if waveform == "cw":
        logger.info("TX buffer — CW (constant envelope, scale=%d)", DAC_HALF_SCALE)
        return np.full(buffer_size, DAC_HALF_SCALE + 0j, dtype=np.complex64)

    if abs(offset_hz) >= f_s_hz / 2.5:          # Sean's 4/5 rule
        raise ValueError(
            f"offset_hz={offset_hz} Hz is not below 0.4 x f_s (f_s/2.5 = {f_s_hz / 2.5} Hz, "
            f"4/5 of the Nyquist frequency). Reduce the offset or increase the sampling rate."
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
        DAC_HALF_SCALE,
    )
    t_s = np.arange(buffer_size, dtype=np.float64) / f_s_hz
    waveform_samples = DAC_HALF_SCALE * np.exp(1j * 2 * np.pi * offset_hz * t_s)
    return waveform_samples.astype(np.complex64)


def snap_tx_offset_hz(requested_offset_hz: float, f_s_hz: float, buffer_size: int) -> float:
    """Nearest multiple of ``f_s / buffer_size`` (15.26 Hz at 65536 / 1 MS/s): whole
    number of periods per cyclic TX buffer.  ``ValueError`` if it rounds to 0."""
    grid_hz = f_s_hz / buffer_size
    n_periods = round(requested_offset_hz / grid_hz)
    if n_periods == 0:
        raise ValueError(
            f"offset {requested_offset_hz} Hz is too small: the smallest possible "
            f"offset is {grid_hz:.4f} Hz (f_s / buffer_size)."
        )
    return n_periods * grid_hz


def effective_tx_offset_hz(cfg: dict[str, Any]) -> float:
    """TX offset actually transmitted, snapped (Hz; 0 unless ``"cw_offset"``).
    The receiver must use this value too (bug B1)."""
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
    rx_kernel_buffers: int = 4,
):
    """Connect, configure (manual gains, RF bandwidth = *f_s_hz*) and start the cyclic TX.

    *tx_buffer* is already scaled (:func:`cw_tx_buffer`); *rx_kernel_buffers*
    buffers are queued while the host is busy (margin before losses).
    ``RuntimeError`` if ``pyadi-iio`` is missing or the Pluto unreachable.
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
    # Must be set before the first rx(), which creates the RX buffer.
    sdr._rxadc.set_kernel_buffers_count(int(rx_kernel_buffers))
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
    """Clear the sticky bits of the RX status register."""
    sdr._rxadc.reg_write(RX_STATUS_REGISTER, 0x6)


def read_and_clear_rx_overflow(sdr) -> bool:
    """``True`` if RX samples were dropped since the last call (bit cleared after reading)."""
    status = sdr._rxadc.reg_read(RX_STATUS_REGISTER)
    if status & RX_OVERFLOW_BIT:
        sdr._rxadc.reg_write(RX_STATUS_REGISTER, status)
        return True
    return False


def check_saturation(frame: np.ndarray, frame_index: int) -> None:
    """Warn when an RX buffer exceeds 80 % of the ADC full scale (lower the gains)."""
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
