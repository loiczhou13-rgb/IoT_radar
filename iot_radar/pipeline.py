"""Micro-Doppler streaming pipeline: IQ source → decimation → clutter → STFT → detection.

:func:`streaming_frame_generator` assembles the DSP bricks of
:mod:`iot_radar.dsp` around an IQ source of :mod:`iot_radar.acquisition` and
yields one result dictionary per STFT column.  :func:`build_context` gathers
the static quantities shown by the dashboard.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Generator

import numpy as np

from iot_radar.acquisition.pluto import generate_tx_buffer, stream_pluto
from iot_radar.acquisition.sources import stream_simulation
from iot_radar.dsp.clutter import ClutterFilter
from iot_radar.dsp.decimation import Decimator
from iot_radar.dsp.detection import detect_presence_column
from iot_radar.dsp.spectral import compute_single_column, frequency_axis, get_window
from iot_radar.physics import SPEED_OF_LIGHT, compute_range

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def resolve_f_offset(cfg: dict[str, Any]) -> float:
    """Return the effective baseband offset (Hz), 0 if not in cw_offset mode."""
    emi = cfg.get("emission", {})
    if emi.get("mode") == "cw_offset":
        return float(emi.get("f_offset", 0.0))
    return 0.0


def auto_skip_warmup(
    clu_cfg: dict[str, Any],
    f_s_dec: float,
    hop: int,
) -> int:
    """Compute a sensible warm-up length from the active clutter filter.

    The clutter filter has a transient of ~3·τ.  We translate it into a
    number of STFT hops and round up.  The user can still override via
    ``spectrogramme.skip_warmup`` in the config.
    """
    mode = clu_cfg.get("mode", "butterworth")
    if mode in ("iir", "mean"):
        alpha = float(clu_cfg.get("alpha", 0.9999))
        tau_s = 1.0 / max((1.0 - alpha) * f_s_dec, 1e-12)
    elif mode == "butterworth":
        f_cut = float(clu_cfg.get("butterworth_cutoff", 0.05))
        tau_s = 1.0 / (2.0 * math.pi * max(f_cut, 1e-6))
    else:
        tau_s = 0.0

    hop_s = hop / f_s_dec
    return int(math.ceil(3.0 * tau_s / hop_s)) if hop_s > 0 else 0


# ------------------------------------------------------------------
# Streaming pipeline
# ------------------------------------------------------------------

def open_iq_stream(
    cfg: dict[str, Any],
    simulation: bool,
) -> Generator[np.ndarray, None, None]:
    """Return an infinite IQ-buffer generator (hardware or simulation)."""
    sdr = cfg["sdr"]
    sim = cfg["simulation"]
    f_off = resolve_f_offset(cfg)

    if simulation or sim.get("enable", False):
        logger.info("Mode simulation continu activé (f_offset=%.1f Hz)", f_off)
        return stream_simulation(
            f_c=sdr["f_c"],
            f_s=sdr["f_s"],
            buffer_size=sdr["buffer_size"],
            fv=sim["fv"],
            D_mm=sim["D_mm"],
            snr_dB=sim["snr_dB"],
            f_offset=f_off,
            clutter_amplitude=sim.get("clutter_amplitude", 100.0),
        )

    logger.info("Mode matériel continu — connexion au PlutoSDR")
    tx_buffer = generate_tx_buffer(
        mode=cfg["emission"]["mode"],
        buffer_size=sdr["buffer_size"],
        f_s=sdr["f_s"],
        f_offset=f_off,
    )
    return stream_pluto(
        uri=sdr["uri"],
        f_c=sdr["f_c"],
        f_s=sdr["f_s"],
        rx_gain=sdr["rx_gain"],
        tx_gain=sdr["tx_gain"],
        buffer_size=sdr["buffer_size"],
        tx_buffer=tx_buffer,
    )


def streaming_frame_generator(
    cfg: dict[str, Any],
    simulation: bool,
    *,
    decimated_iq_chunks: list[np.ndarray] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Acquire → decimate → clutter → FFT → detect → yield, indefinitely.

    Parameters
    ----------
    cfg, simulation
        Identique aux autres appels du pipeline.
    decimated_iq_chunks
        Si fourni, chaque bloc ``iq`` après décimation et filtre clutter est
        recopié dans cette liste (*debug / enregistrement .wav hors ligne*).
    """
    sdr_cfg = cfg["sdr"]
    dec_cfg = cfg["decimation"]
    clu_cfg = cfg["clutter"]
    spec_cfg = cfg["spectrogramme"]
    win_cfg = cfg["windowing"]
    det_cfg = cfg["detection"]

    f_s = float(sdr_cfg["f_s"])
    n_fft = int(spec_cfg["n_fft"])
    overlap = float(spec_cfg["overlap"])
    hop = max(1, int(n_fft * (1.0 - overlap)))

    do_decimate = dec_cfg.get("enable", True)
    D = int(dec_cfg["D"]) if do_decimate else 1
    f_max_utile = float(dec_cfg["f_max_utile"])

    f_off = resolve_f_offset(cfg)
    bande_resp_bb = tuple(det_cfg["bande_respiration"])
    bande_ref_bb = tuple(det_cfg["bande_reference"])

    f_max_eff = max(f_max_utile, abs(f_off) + bande_ref_bb[1])
    decimator = Decimator(f_s=f_s, D=D, f_max_utile=f_max_eff)
    f_s_dec = decimator.f_s_out

    clutter_filter = ClutterFilter(
        mode=clu_cfg["mode"],
        fs=f_s_dec,
        alpha=float(clu_cfg["alpha"]),
        butterworth_order=int(clu_cfg["butterworth_order"]),
        butterworth_cutoff=float(clu_cfg["butterworth_cutoff"]),
    )
    window = get_window(win_cfg["mode"], n_fft)

    w = float(det_cfg["w"])
    alpha_alert = float(det_cfg["alpha"])
    p_value_decades = float(det_cfg["p_value_decades"])
    acf_floor = float(det_cfg["acf_floor"])
    acf_good = float(det_cfg["acf_good"])

    acf_buffer_seconds = float(det_cfg["acf_buffer_seconds"])
    acf_len = max(1, int(round(acf_buffer_seconds * f_s_dec)))

    user_warmup = spec_cfg.get("skip_warmup")
    auto_warmup = auto_skip_warmup(clu_cfg, f_s_dec, hop)
    skip_warmup = int(user_warmup) if user_warmup is not None else auto_warmup

    buf = np.empty(0, dtype=np.complex64)
    iq_stream = open_iq_stream(cfg, simulation)

    logger.info(
        "Pipeline streaming — n_fft=%d, hop=%d, f_s_dec=%.1f Hz, "
        "skip_warmup=%d (auto=%d), f_offset=%.1f Hz",
        n_fft,
        hop,
        f_s_dec,
        skip_warmup,
        auto_warmup,
        f_off,
    )
    logger.info(
        "Bandes (relatives à la porteuse) — respiration=%.2f–%.2f Hz, "
        "référence=%.2f–%.2f Hz, centre spectral f_offset=%.1f Hz",
        bande_resp_bb[0],
        bande_resp_bb[1],
        bande_ref_bb[0],
        bande_ref_bb[1],
        f_off,
    )

    phi_iq_hist = np.empty(0, dtype=np.complex128)
    n_dec_seen = 0

    frame_counter = 0

    for raw_buf in iq_stream:
        iq_dec = decimator(raw_buf)
        iq_filt = clutter_filter(iq_dec)

        if decimated_iq_chunks is not None:
            decimated_iq_chunks.append(
                np.asarray(iq_filt, dtype=np.complex64).copy()
            )

        n_chunk = len(iq_filt)
        if f_off != 0.0:
            k = np.arange(n_chunk, dtype=np.float64) + n_dec_seen
            demod = np.exp(-1j * 2.0 * np.pi * f_off * k / f_s_dec)
            iq_demod = iq_filt.astype(np.complex128) * demod
        else:
            iq_demod = iq_filt.astype(np.complex128)
        n_dec_seen += n_chunk
        phi_iq_hist = np.concatenate((phi_iq_hist, iq_demod))
        if len(phi_iq_hist) > acf_len:
            phi_iq_hist = phi_iq_hist[-acf_len:]

        buf = np.concatenate((buf, np.asarray(iq_filt, dtype=np.complex64)))

        while len(buf) >= n_fft:
            segment = buf[:n_fft].copy()
            buf = buf[hop:]
            frame_counter += 1

            if frame_counter <= skip_warmup:
                logger.debug(
                    "Warm-up : trame %d/%d ignorée", frame_counter, skip_warmup,
                )
                continue

            col = compute_single_column(segment, f_s_dec, window)

            phi_hist = np.unwrap(np.angle(phi_iq_hist))

            score_presence, p_value_f, acf_peak, fv_estimated = (
                detect_presence_column(
                    col_db=col.col_db,
                    f_hz=col.f_hz,
                    phi_buffer=phi_hist,
                    f_s=f_s_dec,
                    bande_respiration=bande_resp_bb,
                    bande_reference=bande_ref_bb,
                    f_center=f_off,
                    w=w,
                    p_value_decades=p_value_decades,
                    acf_floor=acf_floor,
                    acf_good=acf_good,
                )
            )
            alert = bool(p_value_f < alpha_alert)

            yield {
                "spectre_colonne": col.col_db,
                "score_presence":  score_presence,
                "p_value_f":       p_value_f,
                "acf_peak":        acf_peak,
                "fv_estimated":    fv_estimated,
                "n_trame":         frame_counter,
                "detection":       alert,
            }


# ------------------------------------------------------------------
# Build context for the dashboard
# ------------------------------------------------------------------

def build_context(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the dashboard context dict from config alone."""
    sdr = cfg["sdr"]
    emi = cfg["emission"]
    dec_cfg = cfg["decimation"]
    spec_cfg = cfg["spectrogramme"]
    det_cfg = cfg["detection"]

    f_s = float(sdr["f_s"])
    f_c = float(sdr["f_c"])
    D = int(dec_cfg["D"]) if dec_cfg.get("enable", True) else 1
    f_s_dec = f_s / D
    n_fft = int(spec_cfg["n_fft"])

    f_hz = frequency_axis(n_fft, f_s_dec)

    f_off = resolve_f_offset(cfg)
    tx_buffer = generate_tx_buffer(
        mode=emi["mode"],
        buffer_size=sdr["buffer_size"],
        f_s=f_s,
        f_offset=f_off,
    )
    tx_spectrum = np.fft.fftshift(np.fft.fft(tx_buffer, n=len(tx_buffer)))
    eps = 1e-12
    spectre_tx_db = 20.0 * np.log10(np.abs(tx_spectrum) + eps).astype(np.float64)
    f_hz_tx = frequency_axis(len(tx_buffer), f_s)

    R_min, R_max = compute_range(cfg)

    df_hz = f_s_dec / n_fft
    wavelength = SPEED_OF_LIGHT / f_c
    dv_mps = df_hz * wavelength / 2.0

    clutter_mode = cfg["clutter"]["mode"]
    
    bande_resp_bb = list(det_cfg["bande_respiration"])
    B_eff_hz = float(cfg["bilan_liaison"]["B_eff_hz"])

    return {
        "f_hz": f_hz,
        "f_s_dec": f_s_dec,
        "spectre_tx_db": spectre_tx_db,
        "f_hz_tx": f_hz_tx,
        "R_min_m": R_min,
        "R_max_m": R_max,
        "df_hz": df_hz,
        "dv_mps": dv_mps,
        "n_fft": n_fft,
        "clutter_mode": clutter_mode,
        "bande_resp": bande_resp_bb,
        "B_eff_hz": B_eff_hz,
    }
