"""
IQ acquisition and TX streaming with PlutoSDR via pyadi-iio.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .emission import scale_for_pyadi_tx

logger = logging.getLogger(__name__)

# ADC full-scale magnitude used for saturation checks (Pluto IQ commonly ±2048).
_ADC_FULL_SCALE: float = 2048.0
_SATURATION_FRACTION: float = 0.8
_SATURATION_THRESHOLD: float = _ADC_FULL_SCALE * _SATURATION_FRACTION


@dataclass
class AcquisitionResult:
    """Container for IQ data and acquisition metadata."""

    iq: np.ndarray
    """Complex baseband samples, shape ``(n_frames * buffer_size,)``."""

    duration_s: float
    """Total span of concatenated IQ (unit: s)."""

    f_s: float
    """Sample rate used for this capture (unit: Hz)."""

    f_c: float
    """Carrier frequency (unit: Hz)."""


def _check_saturation(iq_raw_scaled: np.ndarray) -> None:
    """
    Warn if estimated integer IQ magnitudes exceed 80% of full scale.

    Parameters
    ----------
    iq_raw_scaled
        Samples normalized by ``2**11`` as returned from :func:`rx_to_normalized`.

    Notes
    -----
    Physical note: saturated ADC bins flatten the phase trajectory and wipe
    micro-Doppler structure; reduce RX gain or TX power if this triggers.
    """
    mag = np.abs(iq_raw_scaled) * (2.0**11)
    peak = float(np.max(mag)) if mag.size else 0.0
    if peak > _SATURATION_THRESHOLD:
        logger.warning(
            "Possible ADC saturation: max |IQ| ≈ %.0f (%.0f%% of full scale ±%.0f). "
            "Reduce rx_gain or tx_gain.",
            peak,
            100.0 * peak / _ADC_FULL_SCALE,
            _ADC_FULL_SCALE,
        )


def rx_to_normalized(rx: np.ndarray) -> np.ndarray:
    """
    Convert Pluto ``rx()`` output to float complex with RMS near unity scale.

    Parameters
    ----------
    rx
        Raw complex array from ``adi.Pluto.rx()`` (driver-dependent scaling).

    Returns
    -------
    numpy.ndarray
        Complex64 1-D array with the same length as ``rx``.

    Examples
    --------
    >>> rx_to_normalized(np.array([2048 + 0j], dtype=np.complex64)).real
    array([1.], dtype=float32)

    Notes
    -----
    Physical note: consistent scaling makes downstream thresholds comparable
    across sessions; absolute calibration is not required for Doppler products.
    """
    x = np.asarray(rx, dtype=np.complex64)
    return (x / (2.0**11)).astype(np.complex64)


def acquire_pluto(
    uri: str,
    f_c: float,
    f_s: float,
    rx_gain_db: float,
    tx_gain_db: float,
    buffer_size: int,
    n_frames: int,
    tx_iq: np.ndarray,
) -> AcquisitionResult:
    """
    Configure PlutoSDR, start cyclic TX, and capture ``n_frames`` RX buffers.

    Parameters
    ----------
    uri
        libiio URI, e.g. ``"ip:192.168.2.1"`` or ``"usb:"`` (unitless).
    f_c
        LO / carrier frequency (unit: Hz, typical ``2.4e9``).
    f_s
        IQ sample rate (unit: Hz, typical ``2.0e6``).
    rx_gain_db
        Manual RX gain in dB (typical ``40``).
    tx_gain_db
        TX hardware gain in dB (often negative attenuation, typical ``-20``).
    buffer_size
        Samples per ``rx()`` call (typical ``16384``).
    n_frames
        Number of buffers to concatenate (typical ``200``).
    tx_iq
        Cyclic TX buffer (complex64), peak nominally ``2**14`` before scaling.

    Returns
    -------
    AcquisitionResult
        Concatenated IQ and timing metadata.

    Examples
    --------
    >>> # result = acquire_pluto("ip:192.168.2.1", 2.4e9, 2e6, 40, -20, 256, 4, iq)

    Notes
    -----
    Physical note: coherent CW illumination with shared LO on TX/RX preserves
    phase of the echo so that slow chest motion modulates the baseband phase.
    """
    try:
        import adi  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "pyadi-iio is not installed. Install with `pip install pyadi-iio` "
            "and ensure libiio is available on the system."
        ) from exc

    buffer_size = int(buffer_size)
    n_frames = int(n_frames)
    if buffer_size < 1 or n_frames < 1:
        raise ValueError("buffer_size and n_frames must be positive integers.")

    logger.info(
        "Connecting to PlutoSDR uri=%s f_c=%.3e Hz f_s=%.3e Hz buffer=%d frames=%d",
        uri,
        f_c,
        f_s,
        buffer_size,
        n_frames,
    )

    try:
        sdr: Any = adi.Pluto(uri)
    except Exception as exc:
        logger.error("PlutoSDR connection failed: %s", exc)
        raise RuntimeError(
            "Could not open PlutoSDR. Check USB/Ethernet cable, power, "
            "URI (e.g. ip:192.168.2.1), and that libiio sees the device "
            "(libiio-utils / iio_info)."
        ) from exc

    sdr.sample_rate = int(f_s)
    sdr.rx_lo = int(f_c)
    sdr.tx_lo = int(f_c)
    sdr.rx_buffer_size = buffer_size
    sdr.rx_gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = int(rx_gain_db)
    sdr.tx_gain_control_mode_chan0 = "manual"
    sdr.tx_hardwaregain_chan0 = int(tx_gain_db)

    tx_scaled = scale_for_pyadi_tx(np.asarray(tx_iq, dtype=np.complex64))
    sdr.tx_destroy_buffer()
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_scaled)

    chunks: list[np.ndarray] = []
    t0 = time.perf_counter()
    for k in range(n_frames):
        raw = sdr.rx()
        x = rx_to_normalized(np.asarray(raw, dtype=np.complex64))
        _check_saturation(x)
        chunks.append(x)
    elapsed = time.perf_counter() - t0

    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

    iq = np.concatenate(chunks, axis=0).astype(np.complex64)
    duration_s = float(iq.shape[0]) / float(f_s)
    logger.info(
        "Acquisition finished: %.3f s of IQ (nominal), loop wall time %.3f s",
        duration_s,
        elapsed,
    )
    return AcquisitionResult(iq=iq, duration_s=duration_s, f_s=float(f_s), f_c=float(f_c))


def synthesize_iq(
    f_s: float,
    f_c: float,
    n_samples: int,
    fv_hz: float,
    displacement_m: float,
    snr_db: float,
    clutter_to_micro_ratio: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate synthetic baseband IQ with breathing line, strong clutter, and noise.

    The micro-Doppler tone is simplified as ``J1(m) * exp(j*2*pi*fv*t)`` on top
    of a static clutter phasor and complex Gaussian noise, matching the prompt.

    Parameters
    ----------
    f_s
        Sample rate (unit: Hz).
    f_c
        Carrier used only to compute wavelength (unit: Hz).
    n_samples
        Total IQ samples (unit: samples).
    fv_hz
        Breathing frequency (unit: Hz, typical ``0.3``).
    displacement_m
        Peak chest displacement (unit: m, e.g. ``0.010`` for 10 mm).
    snr_db
        Target SNR of the micro-Doppler tone vs additive noise in linear domain
        after clutter is *not* counted as noise (typical ``20``).
    clutter_to_micro_ratio
        Complex amplitude ratio ``|clutter| / |micro_line|`` (typical ``100``).
    rng
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Complex64 array of shape ``(n_samples,)``.

    Examples
    --------
    >>> x = synthesize_iq(2e6, 2.4e9, 10000, 0.3, 0.01, 20.0, rng=np.random.default_rng(0))
    >>> x.shape
    (10000,)

    Notes
    -----
    Physical note: the full phase-modulated echo ``exp(j*m*sin(2*pi*fv*t))``
    produces Bessel sidebands at ``±k*fv``; using the first sideband amplitude
    ``J1(m)`` isolates the dominant breathing line when modulation index
    ``m = 4*pi*D/lambda`` is modest.
    """
    from scipy.special import jv  # local import keeps module import light

    rng = rng or np.random.default_rng()
    n_samples = int(n_samples)
    t = np.arange(n_samples, dtype=np.float64) / float(f_s)

    c_light = 299792458.0
    wavelength = c_light / float(f_c)
    m = 4.0 * np.pi * float(displacement_m) / wavelength
    j1 = float(jv(1, m))
    micro = j1 * np.exp(1j * 2.0 * np.pi * float(fv_hz) * t)

    clutter = clutter_to_micro_ratio * np.exp(1j * float(rng.uniform(-np.pi, np.pi)))
    clutter_vec = np.full(n_samples, clutter, dtype=np.complex128)

    signal_power = float(np.mean(np.abs(micro) ** 2))
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_sigma = np.sqrt(noise_power / 2.0)
    noise = (noise_sigma * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))).astype(
        np.complex128
    )

    x = clutter_vec + micro.astype(np.complex128) + noise
    return x.astype(np.complex64)
