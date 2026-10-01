"""IQ sources: every place the samples can come from, behind one interface.

Place in the chain: first step.  The pipeline reads consecutive
:class:`Block` objects from a *source* and does not know whether they come
from the PlutoSDR (:class:`PlutoSource`), from the numerical simulation
(:class:`CWSimulationSource`) or from a recorded session
(:class:`ReplaySource`).

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
    ADC_FULL_SCALE,
    check_saturation,
    clear_rx_overflow,
    cw_tx_buffer,
    effective_tx_offset_hz,
    open_pluto,
    read_and_clear_rx_overflow,
)
from iot_radar.acquisition.recording import SessionReader
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

    kind: str
    """``"pluto"``, ``"simulation"`` or ``"replay"`` (stored in the sessions)."""

    firmware_version: str
    """Firmware of the receiver (``"simulation"`` for the simulated source)."""

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
    kind: str = "pluto"

    def __init__(
        self,
        uri: str,
        f_c_hz: float,
        f_s_hz: float,
        rx_gain_db: float,
        tx_gain_db: float,
        buffer_size: int,
        tx_buffer: np.ndarray,
    ) -> None:
        self._sdr = open_pluto(uri, f_c_hz, f_s_hz, rx_gain_db, tx_gain_db, buffer_size, tx_buffer)
        self.sample_rate_hz = float(f_s_hz)
        self.firmware_version = _pluto_firmware_version(self._sdr)
        self._n_blocks = 0
        self._n_samples = 0
        self._overflow_check_available = True
        try:
            clear_rx_overflow(self._sdr)
        except Exception as exc:
            self._disable_overflow_check(exc)
        logger.info(
            "PlutoSDR streaming — %d-sample buffers at %.0f Hz",
            buffer_size,
            f_s_hz,
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
        logger.info("PlutoSDR streaming stopped after %d buffers", self._n_blocks)

    def _overflow_since_last_block(self) -> bool:
        if not self._overflow_check_available:
            return False
        try:
            overflow = read_and_clear_rx_overflow(self._sdr)
        except Exception as exc:
            self._disable_overflow_check(exc)
            return False
        if overflow:
            logger.warning("PlutoSDR: RX samples lost before buffer %d", self._n_blocks)
        return overflow

    def _disable_overflow_check(self, exc: Exception) -> None:
        self._overflow_check_available = False
        logger.warning("RX loss detection unavailable (%s) — overflow always False", exc)


def _pluto_firmware_version(sdr) -> str:
    """Firmware version reported by the libiio context (``""`` if unknown)."""
    try:
        return str(sdr._ctx.attrs.get("fw_version", ""))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

SCENE_STATES: tuple[str, ...] = ("breathing", "empty", "motion", "apnea")
"""States of the simulated scene (they are also annotation labels)."""


def _parse_timeline(timeline: list | None) -> tuple[np.ndarray, np.ndarray]:
    """``[[t_start_s, state], ...]`` -> (start times, state indices), sorted by time."""
    if not timeline:
        return np.zeros(0), np.zeros(0, dtype=np.int8)
    entries = sorted((float(start_s), str(state)) for start_s, state in timeline)
    for _, state in entries:
        if state not in SCENE_STATES:
            raise ValueError(f"Unknown scene state '{state}', use {SCENE_STATES}.")
    starts_s = np.array([start_s for start_s, _ in entries])
    states = np.array([SCENE_STATES.index(state) for _, state in entries], dtype=np.int8)
    return starts_s, states

class CWSimulationSource:
    """Simulated PlutoSDR reception of a CW / CW-offset radar, block by block.

    Physical model (baseband, in ADC units), for the TX tone
    ``s(t) = exp(j·2π·f_offset·t)`` (``f_offset = 0`` in pure CW)::

        r(t) = C_rx                                  receiver DC offset, at 0 Hz
             + C_static · s(t)                       TX→RX leakage and static clutter
             + A · exp(−j·4π·(R0 + d(t))/λ) · s(t)   echo of the breathing person
             + w(t)                                  complex white noise

    where the echoes are multiplied by ``exp(j·2π·δf·t)`` for a residual
    TX/RX local-oscillator offset ``δf``.  Every echo — leakage and walls
    included — is a delayed copy of the TX tone, hence sits at ``+f_offset``
    (bug B2: the former model put the static clutter at 0 Hz, where the
    clutter high-pass removed it perfectly, which never happens on the
    hardware).

    Scene model (:meth:`presence_and_displacement`): the radial chest
    displacement is ``d(t) = D·sin(2π·f_b·t) + D_h·sin(2π·f_h·t)``
    (breathing + heartbeat).  An optional **timeline** changes the scene over
    time, e.g. ``[[0, "breathing"], [60, "empty"], [90, "motion"]]``: each
    entry ``[t_start_s, state]`` holds until the next one, with the states

    * ``breathing`` — person present, breathing (and heartbeat);
    * ``empty`` — nobody in front of the radar (no echo of a person);
    * ``motion`` — person moving: a sway of several centimetres is added;
    * ``apnea`` — person present, holding the breath (heartbeat only).

    Without a timeline the scene is ``breathing`` if *presence* else ``empty``.

    The samples are finally rounded to the integer levels of the 12-bit ADC
    and clipped to ``[-2048, 2047]``, as delivered by the Pluto.

    Parameters
    ----------
    f_c_hz, f_s_hz : float
        Carrier frequency (Hz) and sampling rate (Hz).
    buffer_size : int
        Samples per block.
    tx_offset_hz : float
        Effective baseband TX offset (Hz); 0 in pure CW.
    breath_rate_hz : float
        Breathing frequency ``f_b`` (Hz).
    breath_amplitude_mm : float
        Peak chest displacement ``D`` (mm).
    presence : bool, optional
        ``False`` simulates an empty scene (ignored when *timeline* is given).
    heart_rate_hz, heart_amplitude_mm : float, optional
        Heartbeat frequency (Hz) and peak chest displacement (mm).
    timeline : list or None, optional
        Scene timeline ``[[t_start_s, state], ...]`` (see above).
    motion_amplitude_mm : float, optional
        Amplitude of the body sway during ``motion`` periods (mm).
    target_range_m : float, optional
        Distance ``R0`` of the person (m); sets the phase of its echo.
    target_amplitude : float, optional
        Amplitude ``A`` of the echo of the person (ADC units).
    static_clutter_amplitude : float, optional
        Amplitude of the leakage and static clutter ``|C_static|`` (ADC units).
    receiver_dc_amplitude : float, optional
        Amplitude of the receiver DC offset ``|C_rx|`` (ADC units).
    lo_offset_hz : float, optional
        Residual TX/RX local-oscillator offset ``δf`` (Hz).
    snr_db : float, optional
        Ratio ``A² / noise power`` per sample (dB), before any processing gain.
    seed : int or None, optional
        Seed of the noise generator (``None``: different noise at each run).
    realtime : bool, optional
        If ``True``, :meth:`read_block` waits so that blocks are delivered at
        the real sampling rate, like the hardware (bug B5: an unpaced
        simulation produced far more than 1 s of signal per second of wall
        clock, so recordings and live displays ran too fast).  ``False``
        (default) generates as fast as possible, for tests and offline use.
    """

    n_channels: int = 1
    kind: str = "simulation"
    firmware_version: str = "simulation"

    def __init__(
        self,
        f_c_hz: float,
        f_s_hz: float,
        buffer_size: int,
        tx_offset_hz: float,
        breath_rate_hz: float,
        breath_amplitude_mm: float,
        presence: bool = True,
        heart_rate_hz: float = 1.2,
        heart_amplitude_mm: float = 0.0,
        timeline: list | None = None,
        motion_amplitude_mm: float = 40.0,
        target_range_m: float = 3.0,
        target_amplitude: float = 1.0,
        static_clutter_amplitude: float = 30.0,
        receiver_dc_amplitude: float = 50.0,
        lo_offset_hz: float = 0.0,
        snr_db: float = -25.0,
        seed: int | None = None,
        realtime: bool = False,
    ) -> None:
        self.sample_rate_hz = float(f_s_hz)
        self._buffer_size = int(buffer_size)
        self._tx_offset_hz = float(tx_offset_hz)
        self._wavelength_m = SPEED_OF_LIGHT / f_c_hz
        self._breath_rate_hz = float(breath_rate_hz)
        self._breath_amplitude_m = float(breath_amplitude_mm) * 1e-3
        self._presence = bool(presence)
        self._heart_rate_hz = float(heart_rate_hz)
        self._heart_amplitude_m = float(heart_amplitude_mm) * 1e-3
        self._motion_amplitude_m = float(motion_amplitude_mm) * 1e-3
        self._timeline_starts_s, self._timeline_states = _parse_timeline(timeline)
        self._target_range_m = float(target_range_m)
        self._target_amplitude = float(target_amplitude)
        # Fixed, arbitrary phases: only the geometry of the IQ arc matters.
        self._static_clutter = float(static_clutter_amplitude) * np.exp(1j * 0.7)
        self._receiver_dc = float(receiver_dc_amplitude) * np.exp(-1j * 2.1)
        self._lo_offset_hz = float(lo_offset_hz)
        self._noise_std = self._target_amplitude * 10.0 ** (-float(snr_db) / 20.0) / np.sqrt(2.0)
        self._rng = np.random.default_rng(seed)
        self._sample_idx = 0
        self._realtime = bool(realtime)
        self._next_block_time_s = time.monotonic()

        logger.info(
            "Simulated stream — breathing %.2f Hz, %.1f mm, scene=%s, "
            "SNR per sample %.0f dB, TX offset %.2f Hz, LO offset %.3f Hz",
            breath_rate_hz,
            breath_amplitude_mm,
            timeline if timeline else ("breathing" if self._presence else "empty"),
            snr_db,
            self._tx_offset_hz,
            self._lo_offset_hz,
        )

    def read_block(self) -> Block:
        """Generate the next block of the simulated reception."""
        n = self._buffer_size
        t = (np.arange(n, dtype=np.float64) + self._sample_idx) / self.sample_rate_hz

        tx_tone = np.exp(2j * np.pi * self._tx_offset_hz * t)
        presence, displacement_m = self.presence_and_displacement(t)
        target_phase = -4.0 * np.pi * (self._target_range_m + displacement_m) / self._wavelength_m
        target_echo = self._target_amplitude * presence * np.exp(1j * target_phase)
        echoes = (self._static_clutter + target_echo) * tx_tone
        if self._lo_offset_hz != 0.0:
            echoes = echoes * np.exp(2j * np.pi * self._lo_offset_hz * t)

        noise = self._noise_std * (self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n))
        received = self._receiver_dc + echoes + noise

        samples = self._quantize(received)
        if self._realtime:
            self._wait_for_real_time(n)
        block = Block(samples[np.newaxis, :], self._sample_idx, time.monotonic(), False)
        self._sample_idx += n
        return block

    def scene_states(self, t_s: np.ndarray) -> np.ndarray:
        """Index in :data:`SCENE_STATES` of the scene state at each time *t_s*."""
        t_s = np.asarray(t_s, dtype=np.float64)
        if self._timeline_states.size == 0:
            constant = SCENE_STATES.index("breathing" if self._presence else "empty")
            return np.full(t_s.shape, constant, dtype=np.int8)
        entry = np.searchsorted(self._timeline_starts_s, t_s, side="right") - 1
        states = self._timeline_states[np.clip(entry, 0, None)]
        before_first_entry = SCENE_STATES.index("breathing")
        return np.where(entry >= 0, states, before_first_entry).astype(np.int8)

    def presence_and_displacement(self, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Presence (1 or 0) and radial chest displacement (m) at times *t_s*.

        The displacement is positive away from the radar (an inhalation
        brings the chest closer: negative displacement).
        """
        t_s = np.asarray(t_s, dtype=np.float64)
        states = self.scene_states(t_s)
        breathing_m = np.where(
            states == SCENE_STATES.index("apnea"),
            0.0,
            self._breath_amplitude_m * np.sin(2.0 * np.pi * self._breath_rate_hz * t_s),
        )
        heartbeat_m = self._heart_amplitude_m * np.sin(2.0 * np.pi * self._heart_rate_hz * t_s)
        displacement_m = breathing_m + heartbeat_m
        sway_m = self._motion_amplitude_m * (
            np.sin(2.0 * np.pi * 0.23 * t_s) + 0.6 * np.sin(2.0 * np.pi * 0.71 * t_s + 1.3)
        )
        displacement_m = displacement_m + np.where(states == SCENE_STATES.index("motion"), sway_m, 0.0)
        presence = (states != SCENE_STATES.index("empty")).astype(np.float64)
        return presence, displacement_m

    def chest_displacement_m(self, t_s: np.ndarray) -> np.ndarray:
        """Simulated chest displacement at times *t_s* (m), 0 when nobody is there.

        This is the ground truth stored in the recorded sessions.
        """
        presence, displacement_m = self.presence_and_displacement(t_s)
        return presence * displacement_m

    def scene_annotations(self, n_samples: int) -> list[tuple[int, int, str]]:
        """``(sample_start, sample_count, label)`` of the scene over the first *n_samples*."""
        if self._timeline_states.size == 0:
            label = "breathing" if self._presence else "empty"
            return [(0, n_samples, label)] if n_samples > 0 else []
        # Sample index where each timeline entry starts, within the session.
        starts = [min(n_samples, max(0, int(round(t * self.sample_rate_hz)))) for t in self._timeline_starts_s]
        labels = [SCENE_STATES[int(state)] for state in self._timeline_states]
        if starts[0] > 0:  # before the first entry the person breathes (see scene_states)
            starts = [0] + starts
            labels = ["breathing"] + labels
        stops = starts[1:] + [n_samples]
        return [(start, stop - start, label) for start, stop, label in zip(starts, stops, labels) if stop > start]

    def close(self) -> None:
        """Nothing to release."""

    def _wait_for_real_time(self, n_samples: int) -> None:
        """Sleep until the block of *n_samples* would be complete on hardware."""
        self._next_block_time_s += n_samples / self.sample_rate_hz
        time.sleep(max(0.0, self._next_block_time_s - time.monotonic()))

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """Round I and Q to the integer ADC levels and clip them (complex64)."""
        i = np.clip(np.round(x.real), -ADC_FULL_SCALE, ADC_FULL_SCALE - 1)
        q = np.clip(np.round(x.imag), -ADC_FULL_SCALE, ADC_FULL_SCALE - 1)
        return (i + 1j * q).astype(np.complex64)


# ---------------------------------------------------------------------------
# Replay of a recorded session
# ---------------------------------------------------------------------------

class ReplaySource:
    """Blocks of a recorded HDF5 session, delivered as during the acquisition.

    The blocks follow the ``/blocks`` table of the session: same boundaries,
    same ``overflow`` flags and host times, so the pipeline sees exactly
    what it saw live (and detects the same discontinuities).  Blocks longer
    than *max_block_samples* (e.g. converted legacy recordings, stored as one
    block) are cut into consecutive pieces.

    Parameters
    ----------
    path : pathlib.Path
        Session file written by :class:`iot_radar.acquisition.recording.SessionWriter`.
    speed : float or None, optional
        ``None`` delivers the blocks as fast as possible; ``1.0`` at the real
        sampling rate, ``2.0`` twice as fast...
    max_block_samples : int, optional
        Longest block delivered (samples).
    """

    kind: str = "replay"

    def __init__(self, path, speed: float | None = None, max_block_samples: int = 65536) -> None:
        self._reader = SessionReader(path)
        self.path = self._reader.path
        self.attributes = self._reader.attributes
        self.sample_rate_hz = self._reader.sample_rate_hz
        self.n_channels = self._reader.n_channels
        self.firmware_version = str(self.attributes.get("firmware_version", ""))
        self._speed = speed
        self._pieces = _replay_pieces(self._reader.blocks(), self._reader.n_samples,
                                      self.sample_rate_hz, max_block_samples)
        self._next_piece = 0
        self._next_delivery_s = time.monotonic()
        logger.info(
            "Replaying %s — %d samples at %.0f Hz in %d blocks",
            self.path.name, self._reader.n_samples, self.sample_rate_hz, len(self._pieces),
        )

    def read_block(self) -> Block | None:
        """Next block of the session, or ``None`` at the end."""
        if self._next_piece >= len(self._pieces):
            return None
        start, stop, host_time_s, overflow = self._pieces[self._next_piece]
        self._next_piece += 1
        samples = self._reader.read_iq(start, stop)
        if self._speed:
            self._next_delivery_s += (stop - start) / self.sample_rate_hz / self._speed
            time.sleep(max(0.0, self._next_delivery_s - time.monotonic()))
        return Block(samples, start, host_time_s, overflow)

    def annotations(self) -> list[tuple[int, int, str]]:
        """Annotated segments of the session: ``(sample_start, sample_count, label)``."""
        return self._reader.annotations()

    def close(self) -> None:
        """Close the session file."""
        self._reader.close()


def _replay_pieces(
    blocks: np.ndarray,
    n_samples: int,
    sample_rate_hz: float,
    max_block_samples: int,
) -> list[tuple[int, int, float, bool]]:
    """``(start, stop, host_time_s, overflow)`` of every block to deliver.

    Each row of the ``/blocks`` table spans the samples up to the next row
    (the last one up to the end of ``/iq``).  Rows longer than
    *max_block_samples* are cut; the pieces after the first one get the
    host time of their first sample and no overflow.
    """
    starts = [int(s) for s in blocks["sample_start"]]
    stops = starts[1:] + [n_samples]
    pieces: list[tuple[int, int, float, bool]] = []
    for row, start, stop in zip(blocks, starts, stops):
        for piece_start in range(start, stop, max_block_samples):
            piece_stop = min(piece_start + max_block_samples, stop)
            host_time_s = float(row["host_time_s"]) + (piece_start - start) / sample_rate_hz
            overflow = bool(row["overflow"]) and piece_start == start
            pieces.append((piece_start, piece_stop, host_time_s, overflow))
    return pieces


# ---------------------------------------------------------------------------
# Choice from the configuration
# ---------------------------------------------------------------------------

def open_source(cfg: dict[str, Any], simulation: bool) -> PlutoSource | CWSimulationSource:
    """Open the IQ source described by the configuration.

    The simulation is used when *simulation* is true or when
    ``simulation.enabled`` is set in the configuration; otherwise the
    PlutoSDR is opened and starts transmitting.
    """
    sdr = cfg["sdr"]
    sim = cfg["simulation"]
    tx_offset_hz = effective_tx_offset_hz(cfg)

    if simulation or sim.get("enabled", False):
        logger.info("Simulated source (TX offset %.1f Hz)", tx_offset_hz)
        return CWSimulationSource(
            f_c_hz=sdr["center_frequency_hz"],
            f_s_hz=sdr["sample_rate_hz"],
            buffer_size=sdr["buffer_size"],
            tx_offset_hz=tx_offset_hz,
            breath_rate_hz=sim["breath_rate_hz"],
            breath_amplitude_mm=sim["breath_amplitude_mm"],
            presence=sim.get("presence", True),
            heart_rate_hz=sim.get("heart_rate_hz", 1.2),
            heart_amplitude_mm=sim.get("heart_amplitude_mm", 0.0),
            timeline=sim.get("timeline"),
            motion_amplitude_mm=sim.get("motion_amplitude_mm", 40.0),
            target_range_m=sim.get("target_range_m", 3.0),
            target_amplitude=sim.get("target_amplitude", 1.0),
            static_clutter_amplitude=sim.get("static_clutter_amplitude", 30.0),
            receiver_dc_amplitude=sim.get("receiver_dc_amplitude", 50.0),
            lo_offset_hz=sim.get("lo_offset_hz", 0.0),
            snr_db=sim.get("snr_db", -25.0),
            seed=sim.get("seed"),
            realtime=sim.get("realtime", True),
        )

    logger.info("Hardware source — connecting to the PlutoSDR")
    tx_buffer = cw_tx_buffer(
        waveform=cfg["tx"]["waveform"],
        buffer_size=sdr["buffer_size"],
        f_s_hz=sdr["sample_rate_hz"],
        offset_hz=tx_offset_hz,
    )
    return PlutoSource(
        uri=sdr["uri"],
        f_c_hz=sdr["center_frequency_hz"],
        f_s_hz=sdr["sample_rate_hz"],
        rx_gain_db=sdr["rx_gain_db"],
        tx_gain_db=sdr["tx_gain_db"],
        buffer_size=sdr["buffer_size"],
        tx_buffer=tx_buffer,
    )
