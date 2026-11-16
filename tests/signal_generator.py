"""Test signals of a CW radar, independent of the simulator of the project.

Built from first principles with NumPy only, so that the tests do not check
the processing against the model it was tuned on:

* exact two-way delay ``τ = 2R/c``: ``exp(j2π·f_o·(t − τ)) · exp(−j2π·f_c·τ)``
  at 2.45 GHz;
* breathing rates that are not a whole number of cycles per window;
* echo amplitude varying slowly by up to ±60 %;
* TX→RX leakage far above the echo (at the TX frequency), receiver DC (0 Hz);
* noise density of ``configs/radar.yaml`` (``simulation.snr_db`` at ``sdr.sample_rate_hz``);
* 12-bit quantisation, clipped to [−2048, 2047].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

SPEED_OF_LIGHT_M_S = 299_792_458.0
CARRIER_HZ = 2.45e9
ADC_MIN, ADC_MAX = -2048, 2047
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "radar.yaml"


def config_noise_density_db() -> float:
    """Echo² / noise density (dB·Hz) of the configuration."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return float(cfg["simulation"]["snr_db"]) + 10.0 * np.log10(float(cfg["sdr"]["sample_rate_hz"]))


def breathing(rate_hz: float, peak_to_peak_mm: float) -> Callable[[np.ndarray], np.ndarray]:
    """Chest displacement (m) of a regular breath: an inhalation brings the chest closer."""
    amplitude_m = peak_to_peak_mm * 1e-3 / 2.0

    def displacement_m(t_s: np.ndarray) -> np.ndarray:
        # Slightly asymmetric breath (second harmonic), as a real chest.
        phase = 2.0 * np.pi * rate_hz * t_s
        return -amplitude_m * (np.sin(phase) + 0.15 * np.sin(2.0 * phase + 0.4))

    return displacement_m


def with_apnea(base: Callable[[np.ndarray], np.ndarray], start_s: float, stop_s: float
               ) -> Callable[[np.ndarray], np.ndarray]:
    """*base*, frozen at its value of *start_s* between *start_s* and *stop_s* (breath held)."""

    def displacement_m(t_s: np.ndarray) -> np.ndarray:
        d = base(t_s)
        held = (t_s >= start_s) & (t_s < stop_s)
        frozen = base(np.array([start_s]))[0]
        return np.where(held, frozen, d)

    return displacement_m


def with_motion(base: Callable[[np.ndarray], np.ndarray], start_s: float, stop_s: float,
                amplitude_mm: float = 60.0) -> Callable[[np.ndarray], np.ndarray]:
    """*base* plus a body movement of several centimetres between *start_s* and *stop_s*."""

    def displacement_m(t_s: np.ndarray) -> np.ndarray:
        moving = (t_s >= start_s) & (t_s < stop_s)
        u = np.clip((t_s - start_s) / (stop_s - start_s), 0.0, 1.0)
        sway = amplitude_mm * 1e-3 * (np.sin(2.0 * np.pi * 0.37 * t_s) + 0.7 * np.sin(2.0 * np.pi * 0.93 * t_s))
        return base(t_s) + np.where(moving, sway * np.sin(np.pi * u), 0.0)

    return displacement_m


@dataclass
class RadarScene:
    """One target in front of a CW radar, as received by the ADC.

    ``displacement_m``: chest displacement (m) as a function of time, ``None``
    for an empty scene.  Amplitudes in ADC units; ``amplitude_variation``
    0.6 = ±60 %; ``noise_density_db``: echo² / noise density (dB·Hz).
    """

    sample_rate_hz: float = 20_000.0
    tx_offset_hz: float = 244.140625
    range_m: float = 3.0
    displacement_m: Callable[[np.ndarray], np.ndarray] | None = field(
        default_factory=lambda: breathing(0.27, 8.0))
    echo_amplitude: float = 1.0
    amplitude_variation: float = 0.6
    leakage: complex = 300.0 * np.exp(0.7j)
    receiver_dc: complex = 40.0 * np.exp(-2.1j)
    noise_density_db: float = field(default_factory=config_noise_density_db)
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._noise_rng = np.random.default_rng(self.seed + 1)
        # Slow amplitude variation: two incommensurate tones, peak ±amplitude_variation.
        self._amplitude_phases = rng.uniform(0.0, 2.0 * np.pi, 2)
        snr_per_sample_db = self.noise_density_db - 10.0 * np.log10(self.sample_rate_hz)
        self.noise_std = self.echo_amplitude * 10.0 ** (-snr_per_sample_db / 20.0) / np.sqrt(2.0)

    def echo_amplitude_at(self, t_s: np.ndarray) -> np.ndarray:
        """Echo amplitude (ADC units) at times *t_s*: ``A·(1 + v·(0.6·sin + 0.4·sin))``."""
        a, b = self._amplitude_phases
        variation = 0.6 * np.sin(2.0 * np.pi * 0.031 * t_s + a) + 0.4 * np.sin(2.0 * np.pi * 0.073 * t_s + b)
        return self.echo_amplitude * (1.0 + self.amplitude_variation * variation)

    def samples(self, start: int, count: int) -> np.ndarray:
        """ADC samples ``[start, start + count)`` (complex64, integer I and Q)."""
        t_s = (start + np.arange(count)) / self.sample_rate_hz
        tone = np.exp(2j * np.pi * self.tx_offset_hz * t_s)
        received = self.receiver_dc + self.leakage * tone
        if self.displacement_m is not None:
            delay_s = 2.0 * (self.range_m + self.displacement_m(t_s)) / SPEED_OF_LIGHT_M_S
            echo = (self.echo_amplitude_at(t_s) * np.exp(2j * np.pi * self.tx_offset_hz * (t_s - delay_s))
                    * np.exp(-2j * np.pi * CARRIER_HZ * delay_s))
            received = received + echo
        noise = self.noise_std * (self._noise_rng.standard_normal(count)
                                  + 1j * self._noise_rng.standard_normal(count))
        received = received + noise
        i = np.clip(np.round(received.real), ADC_MIN, ADC_MAX)
        q = np.clip(np.round(received.imag), ADC_MIN, ADC_MAX)
        return (i + 1j * q).astype(np.complex64)


def blocks(scene: RadarScene, duration_s: float, block_size: int = 2048,
           lost: dict[int, int] | None = None, flagged: set[int] | None = None):
    """Blocks ``(samples (1, n), sample_start, overflow)`` of *duration_s* of *scene*.

    *lost*: block index → samples dropped just before it; *flagged*: blocks
    carrying the loss flag.  As on the Pluto, ``sample_start`` counts the
    delivered samples only, and the flag may come on another block than the loss.
    """
    lost, flagged = lost or {}, flagged or set()
    n_blocks = int(duration_s * scene.sample_rate_hz / block_size)
    true_start = 0
    for index in range(n_blocks):
        true_start += lost.get(index, 0)
        yield scene.samples(true_start, block_size)[np.newaxis, :], index * block_size, index in flagged
        true_start += block_size
