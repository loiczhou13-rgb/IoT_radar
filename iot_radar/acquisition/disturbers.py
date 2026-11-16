"""Simulated people moving around the radar, and the masking of the paths they cross.

Simulation only (used by :class:`iot_radar.acquisition.sources.CWSimulationSource`).
Radar at the origin, boresight ``+y``, metres and seconds.  A disturber is a
few point scatterers; one at range ``R`` returns ``A_1m · sqrt(σ_rel) / R²``
with phase ``−4π·R/λ`` (Doppler ``2·v/λ`` ≈ 16·v Hz at 2.45 GHz).  It moves
only within ``[start_s, stop_s)`` (the annotated disturbance) and stands
still before and after: switching an echo on would itself be a step.
Default torso 12.7 at 1 m (1 m², next to the 0.5 m² chest of amplitude 1 at 3 m).

Walker: simplified from V. C. Chen *et al.*, "Micro-Doppler effect in radar:
phenomenon, model, and simulation study", IEEE Trans. AES 42(1), 2006 (torso
at constant velocity, legs and arms swinging at the gait frequency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

BODY_RADIUS_M: float = 0.25
"""Radius of a torso for the masking of the paths it crosses (m)."""

MASK_DEPTH_DB: float = 20.0
"""Attenuation of a path fully masked by a torso (dB)."""

FIELD_OF_VIEW_DEG: float = 60.0
MAX_RANGE_M: float = 15.0
"""A walker counts as a disturbance within ±60° of the boresight and 15 m."""

STRIDE_M: float = 1.4
"""Gait cycle: gait frequency = speed / 1.4 m (0.36–1.07 Hz for 0.5–1.5 m/s)."""


@dataclass
class Scatterers:
    """Scatterers of one disturber: positions ``(n_scatterers, n_samples, 2)`` (m) and
    ``sqrt`` of their relative radar cross-sections."""

    positions_m: np.ndarray
    relative_amplitudes: np.ndarray


def _active(t_s: np.ndarray, start_s: float, stop_s: float) -> np.ndarray:
    return (t_s >= start_s) & (t_s < stop_s)


def _motion_time(t_s: np.ndarray, start_s: float, stop_s: float) -> np.ndarray:
    """Time spent moving at *t_s* (s): 0 before *start_s*, frozen after *stop_s*."""
    return np.clip(np.asarray(t_s, dtype=np.float64), start_s, stop_s) - start_s


def _radial_unit(position_m: np.ndarray) -> np.ndarray:
    """Unit vector from the radar towards *position_m*."""
    return position_m / np.linalg.norm(position_m)


class Walker:
    """Person walking in a straight line from *start_m* at *velocity_m_s* (torso, two legs, two arms).

    *amplitude_at_1m*: echo of the torso at 1 m (ADC units), as for every disturber.
    """

    kind = "walker"
    _RELATIVE_AMPLITUDES = np.sqrt([1.0, 0.25, 0.25, 0.1, 0.1])   # torso, legs, arms
    _SWING_M = np.array([0.0, 0.3, -0.3, -0.15, 0.15])             # along the walking direction
    _SIDE_M = np.array([0.0, 0.1, -0.1, 0.2, -0.2])                # across it

    def __init__(self, start_m, velocity_m_s, start_s: float = 0.0, stop_s: float = np.inf,
                 amplitude_at_1m: float = 12.7) -> None:
        self.start_m = np.asarray(start_m, dtype=np.float64)
        self.velocity_m_s = np.asarray(velocity_m_s, dtype=np.float64)
        self.start_s, self.stop_s = float(start_s), float(stop_s)
        self.amplitude_at_1m = float(amplitude_at_1m)
        speed = float(np.linalg.norm(self.velocity_m_s))
        self.gait_frequency_hz = speed / STRIDE_M
        self._along = self.velocity_m_s / speed if speed > 0 else np.array([0.0, 1.0])
        self._across = np.array([-self._along[1], self._along[0]])

    def torso_m(self, t_s: np.ndarray) -> np.ndarray:
        """Torso position at times *t_s*, shape ``(n, 2)`` (m)."""
        return self.start_m + np.outer(_motion_time(t_s, self.start_s, self.stop_s), self.velocity_m_s)

    def active(self, t_s: np.ndarray) -> np.ndarray:
        """``True`` while the person walks."""
        return _active(np.asarray(t_s), self.start_s, self.stop_s)

    def scatterers(self, t_s: np.ndarray) -> Scatterers:
        """Torso and limbs at times *t_s*."""
        t_s = np.asarray(t_s, dtype=np.float64)
        torso = self.torso_m(t_s)
        swing = np.sin(2.0 * np.pi * self.gait_frequency_hz * _motion_time(t_s, self.start_s, self.stop_s))
        positions = (torso[np.newaxis, :, :]
                     + self._SWING_M[:, np.newaxis, np.newaxis] * swing[np.newaxis, :, np.newaxis] * self._along
                     + self._SIDE_M[:, np.newaxis, np.newaxis] * self._across)
        return Scatterers(positions, self._RELATIVE_AMPLITUDES)

    def disturbance_mask(self, t_s: np.ndarray) -> np.ndarray:
        """``True`` while the walker is in the field of the radar (±60°, 15 m)."""
        torso = self.torso_m(t_s)
        range_m = np.linalg.norm(torso, axis=1)
        angle_deg = np.degrees(np.arctan2(torso[:, 0], torso[:, 1]))
        return self.active(t_s) & (np.abs(angle_deg) <= FIELD_OF_VIEW_DEG) & (range_m <= MAX_RANGE_M)


class Sway:
    """Person standing at *position_m* and swaying radially (≈ 0.8 Hz)."""

    kind = "sway"

    def __init__(self, position_m, amplitude_m: float = 0.03, frequency_hz: float = 0.8, start_s: float = 0.0,
                 stop_s: float = np.inf, amplitude_at_1m: float = 12.7) -> None:
        self.position_m = np.asarray(position_m, dtype=np.float64)
        self.amplitude_m, self.frequency_hz = float(amplitude_m), float(frequency_hz)
        self.start_s, self.stop_s = float(start_s), float(stop_s)
        self.amplitude_at_1m = float(amplitude_at_1m)
        self._radial = _radial_unit(self.position_m)

    def torso_m(self, t_s: np.ndarray) -> np.ndarray:
        """Torso position at times *t_s*, shape ``(n, 2)`` (m)."""
        moving_s = _motion_time(t_s, self.start_s, self.stop_s)
        offset = self.amplitude_m * np.sin(2.0 * np.pi * self.frequency_hz * moving_s)
        return self.position_m + np.outer(offset, self._radial)

    def active(self, t_s: np.ndarray) -> np.ndarray:
        """``True`` during the sway."""
        return _active(np.asarray(t_s), self.start_s, self.stop_s)

    def scatterers(self, t_s: np.ndarray) -> Scatterers:
        """The torso only."""
        return Scatterers(self.torso_m(t_s)[np.newaxis], np.ones(1))

    def disturbance_mask(self, t_s: np.ndarray) -> np.ndarray:
        """``True`` during the sway."""
        return self.active(t_s)


class Gesture:
    """Person standing still (static torso, masking) and moving a hand 30 cm in front of it.

    8 cm at 2 Hz: hand up to 1 m/s, Doppler up to 16 Hz.  The hand returns 0.3 of the torso.
    """

    kind = "gesture"
    _RELATIVE_AMPLITUDES = np.array([1.0, 0.3])   # torso, hand

    def __init__(self, position_m, amplitude_m: float = 0.08, frequency_hz: float = 2.0, start_s: float = 0.0,
                 stop_s: float = np.inf, amplitude_at_1m: float = 12.7) -> None:
        self.position_m = np.asarray(position_m, dtype=np.float64)
        self.amplitude_m, self.frequency_hz = float(amplitude_m), float(frequency_hz)
        self.start_s, self.stop_s = float(start_s), float(stop_s)
        self.amplitude_at_1m = float(amplitude_at_1m)
        self._radial = _radial_unit(self.position_m)

    def torso_m(self, t_s: np.ndarray) -> np.ndarray:
        """Torso position (static), shape ``(n, 2)`` (m)."""
        return np.tile(self.position_m, (np.asarray(t_s).size, 1))

    def active(self, t_s: np.ndarray) -> np.ndarray:
        """``True`` during the gestures."""
        return _active(np.asarray(t_s), self.start_s, self.stop_s)

    def scatterers(self, t_s: np.ndarray) -> Scatterers:
        """Static torso and oscillating hand."""
        t_s = np.asarray(t_s, dtype=np.float64)
        moving_s = _motion_time(t_s, self.start_s, self.stop_s)
        hand_offset = -0.3 + self.amplitude_m * np.sin(2.0 * np.pi * self.frequency_hz * moving_s)
        hand = self.position_m + np.outer(hand_offset, self._radial)
        positions = np.stack([self.torso_m(t_s), hand])
        return Scatterers(positions, self._RELATIVE_AMPLITUDES)

    def disturbance_mask(self, t_s: np.ndarray) -> np.ndarray:
        """``True`` during the gestures."""
        return self.active(t_s)


Disturber = Walker | Sway | Gesture


def disturber_echo(disturber: Disturber, t_s: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Echo of one disturber (ADC units, before the TX tone): ``Σ A_1m·a_i/R_i² · exp(−j4πR_i/λ)``."""
    scatterers = disturber.scatterers(t_s)
    range_m = np.maximum(np.linalg.norm(scatterers.positions_m, axis=-1), 0.3)     # (n_scatterers, n)
    amplitudes = disturber.amplitude_at_1m * scatterers.relative_amplitudes[:, np.newaxis] / range_m ** 2
    echo = np.sum(amplitudes * np.exp(-4j * np.pi * range_m / wavelength_m), axis=0)
    return echo


def path_masking(point_m: np.ndarray, disturbers: list[Disturber], t_s: np.ndarray) -> np.ndarray:
    """Factor in ``[10^(−depth/20), 1]`` of the path radar → *point_m*, per time.

    A torso between them within :data:`BODY_RADIUS_M` of the line of sight
    attenuates by :data:`MASK_DEPTH_DB` (raised-cosine edge over one more
    radius); several torsos multiply.
    """
    point_m = np.asarray(point_m, dtype=np.float64)
    floor = 10.0 ** (-MASK_DEPTH_DB / 20.0)
    factor = np.ones(np.asarray(t_s).size)
    length2 = float(point_m @ point_m)
    for disturber in disturbers:
        torso = disturber.torso_m(t_s)
        along = torso @ point_m / length2                         # 0 at the radar, 1 at the point
        distance = np.linalg.norm(torso - np.outer(along, point_m), axis=1)
        shape = np.clip((2.0 * BODY_RADIUS_M - distance) / BODY_RADIUS_M, 0.0, 1.0)
        shape = 0.5 - 0.5 * np.cos(np.pi * shape)                 # raised cosine: 1 within one radius
        masked = shape * ((along > 0.0) & (along < 1.0))
        factor *= 1.0 - (1.0 - floor) * masked
    return factor


def activity_intervals(disturbers: list[Disturber], n_samples: int, sample_rate_hz: float,
                       grid_hz: float = 10.0) -> list[tuple[int, int, str]]:
    """``(sample_start, sample_count, kind)`` of the disturbances (masks evaluated at *grid_hz*)."""
    duration_s = n_samples / sample_rate_hz
    t_s = np.arange(0.0, duration_s, 1.0 / grid_hz)
    intervals = []
    for disturber in disturbers:
        mask = disturber.disturbance_mask(t_s)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
        for start, stop in zip(edges[::2], edges[1::2]):
            sample_start = int(round(t_s[start] * sample_rate_hz))
            sample_stop = n_samples if stop >= t_s.size else int(round(t_s[stop] * sample_rate_hz))
            intervals.append((sample_start, sample_stop - sample_start, disturber.kind))
    return sorted(intervals)


def disturbers_from_config(entries: list[dict[str, Any]] | None) -> list[Disturber]:
    """Disturbers of ``simulation.disturbers``: ``kind`` (walker, sway, gesture) + class parameters."""
    classes = {"walker": Walker, "sway": Sway, "gesture": Gesture}
    disturbers = []
    for entry in entries or []:
        parameters = dict(entry)
        kind = parameters.pop("kind")
        if kind not in classes:
            raise ValueError(f"Unknown disturber kind '{kind}', use {sorted(classes)}.")
        disturbers.append(classes[kind](**parameters))
    return disturbers
