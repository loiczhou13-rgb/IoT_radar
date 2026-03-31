from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit_power(x: np.ndarray) -> np.ndarray:
    p = float(np.mean(np.abs(x) ** 2) + 1e-12)
    return x / np.sqrt(p)


@dataclass(frozen=True)
class WaveformSpec:
    kind: str = "cw_tone"  # cw_tone | noise | qpsk | chirp
    amplitude: float = 0.5
    tone_hz: float = 10_000.0
    duration_s: float = 0.1

    # qpsk
    sps: int = 8

    # chirp placeholder (baseband)
    chirp_f0_hz: float = -50_000.0
    chirp_f1_hz: float = 50_000.0


def generate(spec: WaveformSpec, fs_hz: float) -> np.ndarray:
    n = max(1, int(round(spec.duration_s * fs_hz)))
    t = np.arange(n, dtype=np.float32) / float(fs_hz)

    if spec.kind == "cw_tone":
        x = np.exp(1j * 2 * np.pi * float(spec.tone_hz) * t)

    elif spec.kind == "noise":
        x = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

    elif spec.kind == "qpsk":
        sps = max(1, int(spec.sps))
        m = max(1, n // sps)
        symbols = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) / np.sqrt(2)
        idx = np.random.randint(0, 4, size=m)
        sym = symbols[idx]
        x = np.repeat(sym, sps)[:n]

    elif spec.kind == "chirp":
        # Placeholder for future FMCW/range work; baseband linear chirp.
        f0 = float(spec.chirp_f0_hz)
        f1 = float(spec.chirp_f1_hz)
        k = (f1 - f0) / max(spec.duration_s, 1e-6)
        phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
        x = np.exp(1j * phase)

    else:
        raise ValueError(f"Unknown waveform kind: {spec.kind!r}")

    x = _unit_power(np.asarray(x, dtype=np.complex64))
    amp = float(np.clip(spec.amplitude, 0.0, 0.99))
    return (x * amp).astype(np.complex64)

