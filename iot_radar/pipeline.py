"""Micro-Doppler streaming pipeline: IQ source → decimation → clutter → STFT → detection.

Place in the chain: the assembly.  :func:`streaming_frame_generator` connects
the DSP bricks of :mod:`iot_radar.dsp` to an IQ source of
:mod:`iot_radar.acquisition` and yields one result dictionary per STFT column;
:func:`build_context` gathers the static quantities shown by the dashboard.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Generator

import numpy as np

from iot_radar.acquisition.pluto import cw_tx_buffer, effective_tx_offset_hz
from iot_radar.acquisition.sources import Source
from iot_radar.dsp.clutter import ClutterFilter
from iot_radar.dsp.decimation import Decimator
from iot_radar.dsp.detection import detect_presence_column
from iot_radar.dsp.spectral import compute_single_column, frequency_axis, get_window
from iot_radar.physics import SPEED_OF_LIGHT, range_interval_m

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def auto_skip_warmup(
    clutter_cfg: dict[str, Any],
    f_s_dec_hz: float,
    hop: int,
) -> int:
    """Number of STFT columns to drop while the clutter filter settles.

    The clutter filter has a transient of about 3·τ (time constant τ); it is
    converted into a number of STFT hops, rounded up.  The configuration can
    override it with ``spectrogram.skip_warmup_frames``.

    Parameters
    ----------
    clutter_cfg : dict
        ``clutter`` section of the configuration.
    f_s_dec_hz : float
        Decimated sampling rate (Hz).
    hop : int
        STFT hop (samples).
    """
    mode = clutter_cfg.get("mode", "butterworth")
    if mode in ("iir", "mean"):
        alpha = float(clutter_cfg.get("alpha", 0.9999))
        tau_s = 1.0 / max((1.0 - alpha) * f_s_dec_hz, 1e-12)
    elif mode == "butterworth":
        cutoff_hz = float(clutter_cfg.get("butterworth_cutoff_hz", 0.05))
        tau_s = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
    else:
        tau_s = 0.0

    hop_s = hop / f_s_dec_hz
    return int(math.ceil(3.0 * tau_s / hop_s)) if hop_s > 0 else 0


# ------------------------------------------------------------------
# Streaming pipeline
# ------------------------------------------------------------------

def streaming_frame_generator(
    cfg: dict[str, Any],
    source: Source,
    *,
    decimated_iq_chunks: list[np.ndarray] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Acquire → decimate → clutter → STFT → detect → yield, until the source ends.

    Parameters
    ----------
    cfg : dict
        Complete configuration (see ``configs/radar.yaml``).
    source : Source
        IQ source (PlutoSDR or simulation); only its first channel is
        processed.  The generator closes it (``source.close()``) when it stops.
    decimated_iq_chunks : list or None, optional
        When given, every decimated and clutter-filtered block is appended to
        this list (to save the decimated IQ of a recording).

    Yields
    ------
    dict
        One frame per STFT column (after the warm-up):

        * ``spectrum_column_db`` — power spectrum (dB), shape ``(n_fft,)``;
        * ``presence_score`` — fused score in [0, 1];
        * ``p_value`` — p-value of the Fisher test;
        * ``acf_peak`` — phase autocorrelation peak;
        * ``breathing_rate_hz`` — rate estimated by the ACF (or ``None``);
        * ``frame_number`` — index of the STFT column (warm-up included);
        * ``alert`` — ``p_value < detection.false_alarm_probability``.
    """
    sdr_cfg = cfg["sdr"]
    decimation_cfg = cfg["decimation"]
    clutter_cfg = cfg["clutter"]
    spectrogram_cfg = cfg["spectrogram"]
    detection_cfg = cfg["detection"]

    f_s_hz = float(sdr_cfg["sample_rate_hz"])
    n_fft = int(spectrogram_cfg["n_fft"])
    overlap = float(spectrogram_cfg["overlap"])
    hop = max(1, int(n_fft * (1.0 - overlap)))

    decimation_enabled = decimation_cfg.get("enabled", True)
    decimation_factor = int(decimation_cfg["factor"]) if decimation_enabled else 1
    max_useful_frequency_hz = float(decimation_cfg["max_useful_frequency_hz"])

    tx_offset_hz = effective_tx_offset_hz(cfg)
    breathing_band_hz = tuple(detection_cfg["breathing_band_hz"])
    reference_band_hz = tuple(detection_cfg["reference_band_hz"])

    # The decimated stream must still contain the TX offset plus the
    # reference band around it.
    max_frequency_hz = max(max_useful_frequency_hz, abs(tx_offset_hz) + reference_band_hz[1])
    decimator = Decimator(
        f_s_hz=f_s_hz, decimation_factor=decimation_factor, max_frequency_hz=max_frequency_hz,
    )
    f_s_dec_hz = decimator.f_s_out_hz

    clutter_filter = ClutterFilter(
        mode=clutter_cfg["mode"],
        f_s_hz=f_s_dec_hz,
        alpha=float(clutter_cfg["alpha"]),
        butterworth_order=int(clutter_cfg["butterworth_order"]),
        butterworth_cutoff_hz=float(clutter_cfg["butterworth_cutoff_hz"]),
    )
    window = get_window(spectrogram_cfg["window"], n_fft)

    spectral_weight = float(detection_cfg["spectral_weight"])
    false_alarm_probability = float(detection_cfg["false_alarm_probability"])
    p_value_decades = float(detection_cfg["p_value_decades"])
    acf_floor = float(detection_cfg["acf_floor"])
    acf_good = float(detection_cfg["acf_good"])

    acf_buffer_s = float(detection_cfg["acf_buffer_s"])
    acf_length = max(1, int(round(acf_buffer_s * f_s_dec_hz)))

    configured_warmup = spectrogram_cfg.get("skip_warmup_frames")
    automatic_warmup = auto_skip_warmup(clutter_cfg, f_s_dec_hz, hop)
    skip_warmup = int(configured_warmup) if configured_warmup is not None else automatic_warmup

    stft_buffer = np.empty(0, dtype=np.complex64)

    logger.info(
        "Streaming pipeline — n_fft=%d, hop=%d, f_s_dec=%.1f Hz, "
        "skip_warmup=%d (auto=%d), TX offset=%.1f Hz",
        n_fft,
        hop,
        f_s_dec_hz,
        skip_warmup,
        automatic_warmup,
        tx_offset_hz,
    )
    logger.info(
        "Bands (relative to the carrier) — breathing=%.2f–%.2f Hz, "
        "reference=%.2f–%.2f Hz, spectral centre=%.1f Hz",
        breathing_band_hz[0],
        breathing_band_hz[1],
        reference_band_hz[0],
        reference_band_hz[1],
        tx_offset_hz,
    )

    demodulated_history = np.empty(0, dtype=np.complex128)
    n_decimated_seen = 0

    frame_counter = 0

    try:
        while True:
            block = source.read_block()
            if block is None:
                break
            raw_samples = block.samples[0]
            iq_decimated = decimator(raw_samples)
            iq_filtered = clutter_filter(iq_decimated)

            if decimated_iq_chunks is not None:
                decimated_iq_chunks.append(
                    np.asarray(iq_filtered, dtype=np.complex64).copy()
                )

            # Bring the echo from +tx_offset_hz back to 0 Hz for the phase ACF.
            n_chunk = len(iq_filtered)
            if tx_offset_hz != 0.0:
                k = np.arange(n_chunk, dtype=np.float64) + n_decimated_seen
                demodulator = np.exp(-1j * 2.0 * np.pi * tx_offset_hz * k / f_s_dec_hz)
                iq_demodulated = iq_filtered.astype(np.complex128) * demodulator
            else:
                iq_demodulated = iq_filtered.astype(np.complex128)
            n_decimated_seen += n_chunk
            demodulated_history = np.concatenate((demodulated_history, iq_demodulated))
            if len(demodulated_history) > acf_length:
                demodulated_history = demodulated_history[-acf_length:]

            stft_buffer = np.concatenate((stft_buffer, np.asarray(iq_filtered, dtype=np.complex64)))

            while len(stft_buffer) >= n_fft:
                segment = stft_buffer[:n_fft].copy()
                stft_buffer = stft_buffer[hop:]
                frame_counter += 1

                if frame_counter <= skip_warmup:
                    logger.debug(
                        "Warm-up: column %d/%d dropped", frame_counter, skip_warmup,
                    )
                    continue

                column = compute_single_column(segment, f_s_dec_hz, window)

                phase_rad = np.unwrap(np.angle(demodulated_history))

                presence_score, p_value, acf_peak, breathing_rate_hz = (
                    detect_presence_column(
                        column_db=column.power_db,
                        f_hz=column.f_hz,
                        phase_rad=phase_rad,
                        f_s_hz=f_s_dec_hz,
                        breathing_band_hz=breathing_band_hz,
                        reference_band_hz=reference_band_hz,
                        f_center_hz=tx_offset_hz,
                        spectral_weight=spectral_weight,
                        p_value_decades=p_value_decades,
                        acf_floor=acf_floor,
                        acf_good=acf_good,
                    )
                )
                alert = bool(p_value < false_alarm_probability)

                yield {
                    "spectrum_column_db": column.power_db,
                    "presence_score": presence_score,
                    "p_value": p_value,
                    "acf_peak": acf_peak,
                    "breathing_rate_hz": breathing_rate_hz,
                    "frame_number": frame_counter,
                    "alert": alert,
                }
    finally:
        source.close()


# ------------------------------------------------------------------
# Static context of the dashboard
# ------------------------------------------------------------------

def build_context(cfg: dict[str, Any]) -> dict[str, Any]:
    """Static quantities shown by the dashboard, computed from the configuration.

    Returns
    -------
    dict
        ``f_hz`` (RX spectrum axis, Hz), ``f_s_dec_hz``, ``tx_spectrum_db`` and
        ``tx_f_hz`` (spectrum of the TX buffer and its axis), ``range_min_m``
        and ``range_max_m`` (link budget), ``frequency_resolution_hz``,
        ``velocity_resolution_m_s``, ``n_fft``, ``clutter_mode``,
        ``breathing_band_hz`` and ``noise_bandwidth_hz``.
    """
    sdr = cfg["sdr"]
    tx_cfg = cfg["tx"]
    decimation_cfg = cfg["decimation"]
    spectrogram_cfg = cfg["spectrogram"]
    detection_cfg = cfg["detection"]

    f_s_hz = float(sdr["sample_rate_hz"])
    f_c_hz = float(sdr["center_frequency_hz"])
    decimation_factor = int(decimation_cfg["factor"]) if decimation_cfg.get("enabled", True) else 1
    f_s_dec_hz = f_s_hz / decimation_factor
    n_fft = int(spectrogram_cfg["n_fft"])

    f_hz = frequency_axis(n_fft, f_s_dec_hz)

    tx_offset_hz = effective_tx_offset_hz(cfg)
    tx_buffer = cw_tx_buffer(
        waveform=tx_cfg["waveform"],
        buffer_size=sdr["buffer_size"],
        f_s_hz=f_s_hz,
        offset_hz=tx_offset_hz,
    )
    tx_spectrum = np.fft.fftshift(np.fft.fft(tx_buffer, n=len(tx_buffer)))
    eps = 1e-12
    tx_spectrum_db = 20.0 * np.log10(np.abs(tx_spectrum) + eps).astype(np.float64)
    tx_f_hz = frequency_axis(len(tx_buffer), f_s_hz)

    range_min_m, range_max_m = range_interval_m(cfg)

    frequency_resolution_hz = f_s_dec_hz / n_fft
    wavelength_m = SPEED_OF_LIGHT / f_c_hz
    velocity_resolution_m_s = frequency_resolution_hz * wavelength_m / 2.0

    return {
        "f_hz": f_hz,
        "f_s_dec_hz": f_s_dec_hz,
        "tx_spectrum_db": tx_spectrum_db,
        "tx_f_hz": tx_f_hz,
        "range_min_m": range_min_m,
        "range_max_m": range_max_m,
        "frequency_resolution_hz": frequency_resolution_hz,
        "velocity_resolution_m_s": velocity_resolution_m_s,
        "n_fft": n_fft,
        "clutter_mode": cfg["clutter"]["mode"],
        "breathing_band_hz": list(detection_cfg["breathing_band_hz"]),
        "noise_bandwidth_hz": float(cfg["link_budget"]["noise_bandwidth_hz"]),
    }
