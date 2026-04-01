"""
Micro-Doppler radar processing pipeline stages.

Each submodule implements one configurable step (emission, acquisition,
decimation, clutter removal, windowing, spectrogram, detection).
"""

from __future__ import annotations

__all__ = [
    "emission",
    "acquisition",
    "decimation",
    "clutter",
    "windowing",
    "spectrogramme",
    "detection",
]
