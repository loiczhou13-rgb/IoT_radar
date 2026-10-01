"""Simulated IQ source for the streaming pipeline.

The streaming pipeline consumes one buffer at a time from a generator: the
PlutoSDR one lives in :mod:`iot_radar.acquisition.pluto`, the numerical
simulation here.
"""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np

from iot_radar.physics import SPEED_OF_LIGHT

logger = logging.getLogger(__name__)


def stream_simulation(
    f_c: float,
    f_s: float,
    buffer_size: int,
    fv: float,
    D_mm: float,
    snr_dB: float,
    f_offset: float = 0.0,
    clutter_amplitude: float = 100.0,
) -> Generator[np.ndarray, None, None]:
    """Yield simulated IQ buffers indefinitely with continuous phase.

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

    Yields
    ------
    numpy.ndarray
        Complex64 vector of shape ``(buffer_size,)``.

    Notes
    -----
    Each call produces the *next* ``buffer_size`` samples of the same
    continuous waveform, maintaining phase continuity across buffers.
    This mimics the real PlutoSDR streaming behaviour.
    """
    wavelength = SPEED_OF_LIGHT / f_c
    D_m = D_mm * 1e-3
    mod_index = 4.0 * np.pi * D_m / wavelength

    noise_power = 10.0 ** (-snr_dB / 10.0)
    noise_std = np.sqrt(noise_power / 2.0)
    rng = np.random.default_rng()

    logger.info(
        "Streaming simulation — fv=%.2f Hz, D=%.1f mm, SNR=%.0f dB (continu)",
        fv,
        D_mm,
        snr_dB,
    )

    sample_idx = 0
    while True:
        t = (np.arange(buffer_size, dtype=np.float64) + sample_idx) / f_s

        phase_mod = mod_index * np.sin(2.0 * np.pi * fv * t)
        carrier = (
            2.0 * np.pi * f_offset * t if f_offset != 0.0 else np.zeros_like(t)
        )
        signal = np.exp(1j * (carrier - phase_mod))

        clutter = clutter_amplitude * np.ones(buffer_size, dtype=np.complex128)
        noise = noise_std * (
            rng.standard_normal(buffer_size)
            + 1j * rng.standard_normal(buffer_size)
        )

        buf = (clutter + signal + noise).astype(np.complex64)
        sample_idx += buffer_size
        yield buf
