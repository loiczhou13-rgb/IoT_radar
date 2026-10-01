"""PlutoSDR (ADALM-PLUTO) transceiver: TX waveform and streaming reception."""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

_ADC_FULL_SCALE: int = 2048
_ADC_SATURATION_RATIO: float = 0.80

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
        If *mode* is not one of ``{"cw", "cw_offset"}``, or if *f_offset*
        violates the Nyquist criterion (``|f_offset| >= f_s / 2``).

    Notes
    -----
    * **CW mode** — baseband samples are a constant ``2**14 + 0j``;
      the RF output is a pure tone at exactly ``f_c``.
    * **CW-offset mode** — baseband samples are
      ``2**14 · exp(j·2π·f_offset·t)``, producing an RF tone at
      ``f_c + f_offset`` and avoiding the DC clutter.

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

    logger.info(
        "Génération du buffer TX — mode CW-offset (f_offset=%.0f Hz, scale=%d)",
        f_offset,
        _DAC_FULL_SCALE,
    )
    t = np.arange(buffer_size, dtype=np.float64) / f_s
    waveform = _DAC_FULL_SCALE * np.exp(1j * 2 * np.pi * f_offset * t)
    return waveform.astype(np.complex64)


def stream_pluto(
    uri: str,
    f_c: float,
    f_s: float,
    rx_gain: float,
    tx_gain: float,
    buffer_size: int,
    tx_buffer: np.ndarray,
) -> Generator[np.ndarray, None, None]:
    """Yield IQ buffers from the PlutoSDR indefinitely.

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

    Yields
    ------
    numpy.ndarray
        Complex64 vector of shape ``(buffer_size,)``.

    Raises
    ------
    RuntimeError
        If the PlutoSDR cannot be reached at *uri*.

    Notes
    -----
    * The TX path runs in **cyclic** mode: the SDR re-plays *tx_buffer*
      indefinitely while we keep receiving.
    * On generator close (``gen.close()`` or context exit) the TX buffer
      is destroyed cleanly so the Pluto is left in a known state.
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

    logger.info(
        "Streaming PlutoSDR — buffer_size=%d à %.0f Hz (continu)",
        buffer_size,
        f_s,
    )

    frame_idx = 0
    try:
        while True:
            frame = sdr.rx()
            _check_saturation(frame, frame_idx)
            yield np.asarray(frame, dtype=np.complex64)
            frame_idx += 1
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        logger.info("Streaming PlutoSDR arrêté après %d trames", frame_idx)


def _check_saturation(frame: np.ndarray, frame_index: int) -> None:
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
