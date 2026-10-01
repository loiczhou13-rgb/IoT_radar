"""IQ sources: every place the samples can come from, behind one interface.

The pipeline reads consecutive :class:`Block` objects from a *source* and
does not know whether they come from the PlutoSDR (:class:`PlutoSource`) or
from the numerical simulation (:class:`CWSimulationSource`).

A block carries what is needed to detect a discontinuity of the stream: the
index of its first sample, the host clock when it was received and an
``overflow`` flag set when samples were lost just before it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from iot_radar.acquisition.pluto import (
    check_saturation,
    clear_rx_overflow,
    generate_tx_buffer,
    open_pluto,
    read_and_clear_rx_overflow,
    resolve_f_offset,
)
from iot_radar.physics import SPEED_OF_LIGHT

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """One block of consecutive IQ samples delivered by a source.

    Attributes
    ----------
    samples : numpy.ndarray
        Complex64 samples, shape ``(n_channels, n_samples)``, in ADC units.
    sample_start : int
        Index of ``samples[:, 0]`` in the stream of samples delivered by the
        source (the next block normally starts at
        ``sample_start + n_samples``).
    host_time_s : float
        Host monotonic clock (s) when the block was received.
    overflow : bool
        ``True`` if samples were lost just before this block.
    """

    samples: np.ndarray
    sample_start: int
    host_time_s: float
    overflow: bool


class Source(Protocol):
    """Anything that delivers consecutive blocks of complex baseband IQ."""

    sample_rate_hz: float
    """Rate of the samples returned by :meth:`read_block` (Hz)."""

    n_channels: int
    """Number of receive channels (rows of ``Block.samples``)."""

    def read_block(self) -> Block | None:
        """Next block, or ``None`` when there is no more data."""

    def close(self) -> None:
        """Release the hardware or the file."""


# ---------------------------------------------------------------------------
# PlutoSDR
# ---------------------------------------------------------------------------

class PlutoSource:
    """IQ blocks received by the PlutoSDR while it transmits *tx_buffer*.

    The device is opened and configured by
    :func:`iot_radar.acquisition.pluto.open_pluto` (same parameters).  Each
    :meth:`read_block` returns one RX buffer.

    **Lost samples.**  When the host does not read fast enough, the FPGA
    drops RX samples and sets a bit of the RX status register.  The bit is
    read (and cleared) after every buffer and reported as ``Block.overflow``.
    If the register cannot be accessed (other firmware), a warning is logged
    once and ``overflow`` stays ``False``.  This check has not been validated
    on hardware yet.
    """

    n_channels: int = 1

    def __init__(
        self,
        uri: str,
        f_c: float,
        f_s: float,
        rx_gain: float,
        tx_gain: float,
        buffer_size: int,
        tx_buffer: np.ndarray,
    ) -> None:
        self._sdr = open_pluto(uri, f_c, f_s, rx_gain, tx_gain, buffer_size, tx_buffer)
        self.sample_rate_hz = float(f_s)
        self._n_blocks = 0
        self._n_samples = 0
        self._overflow_check_available = True
        try:
            clear_rx_overflow(self._sdr)
        except Exception as exc:
            self._disable_overflow_check(exc)
        logger.info(
            "Streaming PlutoSDR — buffer_size=%d à %.0f Hz (continu)",
            buffer_size,
            f_s,
        )

    def read_block(self) -> Block:
        """Receive the next RX buffer (waits until it is full)."""
        frame = self._sdr.rx()
        host_time_s = time.monotonic()
        check_saturation(frame, self._n_blocks)
        overflow = self._overflow_since_last_block()
        samples = np.asarray(frame, dtype=np.complex64)[np.newaxis, :]
        block = Block(samples, self._n_samples, host_time_s, overflow)
        self._n_blocks += 1
        self._n_samples += samples.shape[1]
        return block

    def close(self) -> None:
        """Stop the cyclic transmission (leaves the Pluto in a known state)."""
        try:
            self._sdr.tx_destroy_buffer()
        except Exception:
            pass
        logger.info("Streaming PlutoSDR arrêté après %d trames", self._n_blocks)

    def _overflow_since_last_block(self) -> bool:
        if not self._overflow_check_available:
            return False
        try:
            overflow = read_and_clear_rx_overflow(self._sdr)
        except Exception as exc:
            self._disable_overflow_check(exc)
            return False
        if overflow:
            logger.warning("PlutoSDR : échantillons RX perdus avant la trame %d", self._n_blocks)
        return overflow

    def _disable_overflow_check(self, exc: Exception) -> None:
        self._overflow_check_available = False
        logger.warning("Détection des pertes RX indisponible (%s) — overflow toujours False", exc)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class CWSimulationSource:
    """Simulated IQ buffers with continuous phase, one per :meth:`read_block`.

    Parameters
    ----------
    f_c, f_s : float
        Carrier frequency (Hz) and sampling rate (Hz).
    buffer_size : int
        Samples per buffer.
    fv : float
        Simulated breathing frequency (Hz).
    D_mm : float
        Chest displacement amplitude (mm).
    snr_dB : float
        Target micro-Doppler SNR (dB).
    f_offset : float, optional
        Baseband frequency offset (Hz).  Default is 0.
    clutter_amplitude : float, optional
        Amplitude of the static-clutter component relative to the signal.
        Default 100.0 (40 dB above signal — typical CW radar isolation).
    seed : int or None, optional
        Seed of the noise generator (``None``: different noise at each run).

    Notes
    -----
    Each call produces the *next* ``buffer_size`` samples of the same
    continuous waveform, maintaining phase continuity across buffers.
    This mimics the real PlutoSDR streaming behaviour.
    """

    n_channels: int = 1

    def __init__(
        self,
        f_c: float,
        f_s: float,
        buffer_size: int,
        fv: float,
        D_mm: float,
        snr_dB: float,
        f_offset: float = 0.0,
        clutter_amplitude: float = 100.0,
        seed: int | None = None,
    ) -> None:
        self.sample_rate_hz = float(f_s)
        self._f_s = f_s
        self._buffer_size = buffer_size
        self._fv = fv
        self._f_offset = f_offset
        self._clutter_amplitude = clutter_amplitude

        wavelength = SPEED_OF_LIGHT / f_c
        D_m = D_mm * 1e-3
        self._mod_index = 4.0 * np.pi * D_m / wavelength

        noise_power = 10.0 ** (-snr_dB / 10.0)
        self._noise_std = np.sqrt(noise_power / 2.0)
        self._rng = np.random.default_rng(seed)
        self._sample_idx = 0

        logger.info(
            "Streaming simulation — fv=%.2f Hz, D=%.1f mm, SNR=%.0f dB (continu)",
            fv,
            D_mm,
            snr_dB,
        )

    def read_block(self) -> Block:
        """Generate the next buffer of the simulated signal."""
        buffer_size = self._buffer_size
        t = (np.arange(buffer_size, dtype=np.float64) + self._sample_idx) / self._f_s

        phase_mod = self._mod_index * np.sin(2.0 * np.pi * self._fv * t)
        carrier = (
            2.0 * np.pi * self._f_offset * t if self._f_offset != 0.0 else np.zeros_like(t)
        )
        signal = np.exp(1j * (carrier - phase_mod))

        clutter = self._clutter_amplitude * np.ones(buffer_size, dtype=np.complex128)
        noise = self._noise_std * (
            self._rng.standard_normal(buffer_size)
            + 1j * self._rng.standard_normal(buffer_size)
        )

        buf = (clutter + signal + noise).astype(np.complex64)
        block = Block(buf[np.newaxis, :], self._sample_idx, time.monotonic(), False)
        self._sample_idx += buffer_size
        return block

    def close(self) -> None:
        """Nothing to release."""


# ---------------------------------------------------------------------------
# Choice from the configuration
# ---------------------------------------------------------------------------

def open_source(cfg: dict[str, Any], simulation: bool) -> PlutoSource | CWSimulationSource:
    """Open the IQ source described by the configuration.

    The simulation is used when *simulation* is true or when
    ``simulation.enable`` is set in the configuration; otherwise the
    PlutoSDR is opened and starts transmitting.
    """
    sdr = cfg["sdr"]
    sim = cfg["simulation"]
    f_off = resolve_f_offset(cfg)

    if simulation or sim.get("enable", False):
        logger.info("Mode simulation continu activé (f_offset=%.1f Hz)", f_off)
        return CWSimulationSource(
            f_c=sdr["f_c"],
            f_s=sdr["f_s"],
            buffer_size=sdr["buffer_size"],
            fv=sim["fv"],
            D_mm=sim["D_mm"],
            snr_dB=sim["snr_dB"],
            f_offset=f_off,
            clutter_amplitude=sim.get("clutter_amplitude", 100.0),
            seed=sim.get("seed"),
        )

    logger.info("Mode matériel continu — connexion au PlutoSDR")
    tx_buffer = generate_tx_buffer(
        mode=cfg["emission"]["mode"],
        buffer_size=sdr["buffer_size"],
        f_s=sdr["f_s"],
        f_offset=f_off,
    )
    return PlutoSource(
        uri=sdr["uri"],
        f_c=sdr["f_c"],
        f_s=sdr["f_s"],
        rx_gain=sdr["rx_gain"],
        tx_gain=sdr["tx_gain"],
        buffer_size=sdr["buffer_size"],
        tx_buffer=tx_buffer,
    )
