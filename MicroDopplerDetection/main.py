"""CLI entry point for the micro-Doppler radar pipeline.

Usage
-----
Hardware mode (continuous)::

    python -m MicroDopplerDetection.main --config MicroDopplerDetection/config.yaml

Simulation mode (continuous)::

    python -m MicroDopplerDetection.main --config MicroDopplerDetection/config.yaml --simulation
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Generator

import yaml
import numpy as np

from MicroDopplerDetection.pipeline.emission import generate_tx_buffer
from MicroDopplerDetection.pipeline.acquisition import (
    stream_pluto,
    stream_simulation,
)
from MicroDopplerDetection.pipeline.decimation import decimate_iq
from MicroDopplerDetection.pipeline.clutter import ClutterFilter
from MicroDopplerDetection.pipeline.spectrogramme import (
    ColumnOutput,
    compute_single_column,
)
from MicroDopplerDetection.pipeline.detection import detect_snr_column
from MicroDopplerDetection.pipeline.windowing import get_window
from MicroDopplerDetection.utils.display import DashboardRadar

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT: float = 299_792_458.0
_BOLTZMANN: float = 1.380649e-23
_T0: float = 290.0


def _sigmoid(x: float, center: float, scale: float) -> float:
    """Logistic sigmoid mapping SNR (dB) to probability [0, 1]."""
    z = -(x - center) / scale
    z = max(min(z, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(z))


# ------------------------------------------------------------------
# Radar range equation
# ------------------------------------------------------------------

def _radar_range(
    params: dict[str, Any],
    wavelength: float,
    B_hz: float,
) -> float:
    """Evaluate the radar range equation for one set of link-budget parameters.

    Parameters
    ----------
    params : dict
        Link-budget parameters (P_tx_dBm, G_tx_dBi, G_rx_dBi, sigma_m2,
        NF_dB, L_sys_dB, SNR_min_dB).
    wavelength : float
        Carrier wavelength (m).
    B_hz : float
        Effective processing bandwidth (Hz).

    Returns
    -------
    float
        Detection range in metres.

    Notes
    -----
    Monostatic radar range equation:

    .. math::

        R = \\left(
            \\frac{P_t \\, G_{tx} \\, G_{rx} \\, \\lambda^2 \\, \\sigma}
            {(4\\pi)^3 \\, k_B \\, T_0 \\, B \\, F \\, L \\, SNR_{min}}
        \\right)^{1/4}
    """
    P_tx = 1e-3 * 10.0 ** (params["P_tx_dBm"] / 10.0)
    G_tx = 10.0 ** (params["G_tx_dBi"] / 10.0)
    G_rx = 10.0 ** (params["G_rx_dBi"] / 10.0)
    sigma = params["sigma_m2"]
    F = 10.0 ** (params["NF_dB"] / 10.0)
    L = 10.0 ** (params["L_sys_dB"] / 10.0)
    SNR_min = 10.0 ** (params["SNR_min_dB"] / 10.0)

    numerator = P_tx * G_tx * G_rx * wavelength**2 * sigma
    denominator = (4.0 * np.pi)**3 * _BOLTZMANN * _T0 * B_hz * F * L * SNR_min

    return float((numerator / denominator) ** 0.25)


def _compute_range(
    cfg: dict[str, Any],
    f_s_dec: float,
    n_fft: int,
) -> tuple[float, float]:
    """Compute pessimistic and optimistic detection ranges.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary.
    f_s_dec : float
        Decimated sampling rate (Hz).
    n_fft : int
        FFT size, used to derive processing bandwidth.

    Returns
    -------
    tuple[float, float]
        ``(R_min, R_max)`` — pessimistic and optimistic range in metres.

    Notes
    -----
    Two scenarios are evaluated from ``config.bilan_liaison``:

    * **Optimistic** — free-space propagation, clear line of sight,
      nominal antenna gains, full target RCS.
    * **Pessimistic** — propagation through rubble/debris (high losses),
      degraded antenna patterns, partially obscured target (reduced RCS).

    The operator sees an interval ``[R_min, R_max]`` on the dashboard,
    giving both a lower bound (worst case on the field) and an upper
    bound (best achievable performance).
    """
    bl = cfg.get("bilan_liaison", {})
    f_c = cfg["sdr"]["f_c"]
    wavelength = _SPEED_OF_LIGHT / f_c
    B_hz = bl.get("B_eff_hz", f_s_dec / n_fft)

    defaults_opt = {
        "P_tx_dBm": -13, "G_tx_dBi": 2, "G_rx_dBi": 2,
        "sigma_m2": 0.5, "NF_dB": 4, "L_sys_dB": 3, "SNR_min_dB": 3,
    }
    defaults_pes = {
        "P_tx_dBm": -13, "G_tx_dBi": 0, "G_rx_dBi": 0,
        "sigma_m2": 0.05, "NF_dB": 6, "L_sys_dB": 25, "SNR_min_dB": 3,
    }

    params_opt = {k: bl.get("optimiste", {}).get(k, v) for k, v in defaults_opt.items()}
    params_pes = {k: bl.get("pessimiste", {}).get(k, v) for k, v in defaults_pes.items()}

    R_max = _radar_range(params_opt, wavelength, B_hz)
    R_min = _radar_range(params_pes, wavelength, B_hz)

    logger.info(
        "Portée effective — R_min = %.1f m (pire-cas) / R_max = %.1f m (optimiste) "
        "(B=%.3f Hz, λ=%.3f m)",
        R_min,
        R_max,
        B_hz,
        wavelength,
    )
    return R_min, R_max


# ------------------------------------------------------------------
# Configuration loading
# ------------------------------------------------------------------

def _load_config(path: str) -> dict[str, Any]:
    """Load and return the YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        print(f"ERREUR : fichier de configuration introuvable : {path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def _setup_logging(cfg: dict[str, Any]) -> None:
    """Configure the root logger from the config ``logging`` section."""
    level_name = cfg.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------
# Streaming pipeline
# ------------------------------------------------------------------

def _build_iq_stream(
    cfg: dict[str, Any],
    simulation: bool,
) -> Generator[np.ndarray, None, None]:
    """Return an infinite IQ-buffer generator (hardware or simulation)."""
    sdr = cfg["sdr"]
    emi = cfg["emission"]
    sim = cfg["simulation"]

    if simulation or sim.get("enable", False):
        f_off = emi.get("f_offset", 0.0) if emi.get("mode") == "cw_offset" else 0.0
        logger.info("Mode simulation continu activé (f_offset=%.1f Hz)", f_off)
        return stream_simulation(
            f_c=sdr["f_c"],
            f_s=sdr["f_s"],
            buffer_size=sdr["buffer_size"],
            fv=sim["fv"],
            D_mm=sim["D_mm"],
            snr_dB=sim["snr_dB"],
            f_offset=f_off,
        )

    logger.info("Mode matériel continu — connexion au PlutoSDR")
    tx_buffer = generate_tx_buffer(
        mode=emi["mode"],
        buffer_size=sdr["buffer_size"],
        f_s=sdr["f_s"],
        f_offset=emi.get("f_offset", 0.0),
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


def _streaming_frame_generator(
    cfg: dict[str, Any],
    simulation: bool,
) -> Generator[dict[str, Any], None, None]:
    """Infinite generator: acquire → decimate → clutter → FFT → detect → yield.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary.
    simulation : bool
        Force simulation mode.

    Yields
    ------
    dict[str, Any]
        Frame data for :meth:`DashboardRadar.update_frame`.
    """
    sdr = cfg["sdr"]
    dec_cfg = cfg["decimation"]
    clu_cfg = cfg["clutter"]
    spec_cfg = cfg["spectrogramme"]
    win_cfg = cfg["windowing"]
    det_cfg = cfg["detection"]

    f_s = sdr["f_s"]
    f_c = sdr["f_c"]
    n_fft = spec_cfg["n_fft"]
    overlap = spec_cfg["overlap"]
    hop = int(n_fft * (1.0 - overlap))

    do_decimate = dec_cfg.get("enable", True)
    D = dec_cfg["D"] if do_decimate else 1
    f_max_utile = dec_cfg.get("f_max_utile", 10.0)
    f_s_dec = f_s / D if do_decimate else f_s

    if do_decimate and f_s_dec <= 2.0 * f_max_utile:
        raise ValueError(
            f"Critère de Shannon violé : f_s_dec={f_s_dec:.1f} Hz "
            f"≤ 2×f_max_utile={2.0 * f_max_utile:.1f} Hz."
        )

    clutter_filter = ClutterFilter(
        mode=clu_cfg["mode"],
        alpha=clu_cfg.get("alpha", 0.99),
    )
    window = get_window(win_cfg["mode"], n_fft)

    emi = cfg["emission"]
    f_offset = emi.get("f_offset", 0.0) if emi.get("mode") == "cw_offset" else 0.0

    bande_resp_rel = det_cfg["bande_respiration"]
    bande_ref_rel = det_cfg["bande_reference"]
    bande_resp = (bande_resp_rel[0] + f_offset, bande_resp_rel[1] + f_offset)
    bande_ref = (bande_ref_rel[0] + f_offset, bande_ref_rel[1] + f_offset)

    aff = cfg["affichage"]
    centre_sigmoid = det_cfg.get("centre_sigmoid_dB", 3.0)
    echelle_sigmoid = aff["echelle_sigmoid_dB"]
    seuil_proba = aff.get("seuil_proba", 0.6)
    skip_warmup = spec_cfg.get("skip_warmup", 0)

    ring = deque(maxlen=n_fft)
    iq_stream = _build_iq_stream(cfg, simulation)

    logger.info(
        "Pipeline streaming — n_fft=%d, hop=%d, f_s_dec=%.1f Hz, skip_warmup=%d",
        n_fft,
        hop,
        f_s_dec,
        skip_warmup,
    )
    logger.info(
        "Bandes de détection (f_offset=%.1f Hz) — respiration=%.1f–%.1f Hz, "
        "référence=%.1f–%.1f Hz",
        f_offset,
        bande_resp[0],
        bande_resp[1],
        bande_ref[0],
        bande_ref[1],
    )

    frame_counter = 0
    samples_since_last_fft = 0

    for raw_buf in iq_stream:
        if do_decimate:
            iq_dec, _ = decimate_iq(
                iq=raw_buf, f_s=f_s, D=D, f_max_utile=f_max_utile,
            )
        else:
            iq_dec = raw_buf

        iq_filt = clutter_filter(iq_dec)

        for sample in iq_filt:
            ring.append(sample)
            samples_since_last_fft += 1

            if len(ring) < n_fft:
                continue

            if samples_since_last_fft < hop:
                continue

            samples_since_last_fft = 0
            frame_counter += 1

            if frame_counter <= skip_warmup:
                logger.debug("Warm-up : trame %d/%d ignorée", frame_counter, skip_warmup)
                continue

            segment = np.array(ring, dtype=np.complex64)

            col = compute_single_column(segment, f_s_dec, f_c, window)

            snr_db = detect_snr_column(
                col_db=col.col_db,
                f_hz=col.f_hz,
                bande_respiration=bande_resp,
                bande_reference=bande_ref,
            )
            prob = _sigmoid(snr_db, centre_sigmoid, echelle_sigmoid)
            alert = bool(prob >= seuil_proba)

            yield {
                "signal_iq_dec": segment,
                "spectre_colonne": col.col_db,
                "snr_dB": snr_db,
                "prob": prob,
                "n_trame": frame_counter,
                "detection": alert,
            }


# ------------------------------------------------------------------
# Build context for the dashboard (needs one bootstrap column)
# ------------------------------------------------------------------

def _build_context(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the dashboard context dict from config alone."""
    sdr = cfg["sdr"]
    emi = cfg["emission"]
    dec_cfg = cfg["decimation"]
    spec_cfg = cfg["spectrogramme"]
    det_cfg = cfg["detection"]

    f_s = sdr["f_s"]
    f_c = sdr["f_c"]
    D = dec_cfg["D"] if dec_cfg.get("enable", True) else 1
    f_s_dec = f_s / D
    n_fft = spec_cfg["n_fft"]

    f_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / f_s_dec)).astype(np.float64)

    tx_buffer = generate_tx_buffer(
        mode=emi["mode"],
        buffer_size=sdr["buffer_size"],
        f_s=f_s,
        f_offset=emi.get("f_offset", 0.0),
    )
    tx_spectrum = np.fft.fftshift(np.fft.fft(tx_buffer, n=len(tx_buffer)))
    eps = 1e-12
    spectre_tx_db = 20.0 * np.log10(np.abs(tx_spectrum) + eps).astype(np.float64)
    f_hz_tx = np.fft.fftshift(
        np.fft.fftfreq(len(tx_buffer), d=1.0 / f_s)
    ).astype(np.float64)

    R_min, R_max = _compute_range(cfg, f_s_dec, n_fft)

    df_hz = f_s_dec / n_fft
    wavelength = _SPEED_OF_LIGHT / f_c
    dv_mps = df_hz * wavelength / 2.0

    clutter_mode = cfg.get("clutter", {}).get("mode", "mean")
    f_offset = emi.get("f_offset", 0.0) if emi.get("mode") == "cw_offset" else 0.0
    bande_resp_rel = det_cfg.get("bande_respiration", [0.1, 0.8])
    bande_resp = [bande_resp_rel[0] + f_offset, bande_resp_rel[1] + f_offset]
    bl = cfg.get("bilan_liaison", {})
    B_eff_hz = bl.get("B_eff_hz", df_hz)

    return {
        "f_hz": f_hz,
        "f_s_dec": f_s_dec,
        "f_c": f_c,
        "centre_sigmoid_dB": det_cfg.get("centre_sigmoid_dB", 3.0),
        "spectre_tx_db": spectre_tx_db,
        "f_hz_tx": f_hz_tx,
        "R_min_m": R_min,
        "R_max_m": R_max,
        "df_hz": df_hz,
        "dv_mps": dv_mps,
        "n_fft": n_fft,
        "clutter_mode": clutter_mode,
        "bande_resp": bande_resp,
        "B_eff_hz": B_eff_hz,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Radar micro-Doppler — détection de survivants ensevelis",
    )
    parser.add_argument(
        "--config",
        default="MicroDopplerDetection/config.yaml",
        help="Chemin vers le fichier de configuration YAML (défaut : %(default)s)",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Forcer le mode simulation (pas de PlutoSDR requis)",
    )
    return parser.parse_args()


def main() -> None:
    """Top-level pipeline orchestration (streaming mode)."""
    args = _parse_args()
    cfg = _load_config(args.config)
    _setup_logging(cfg)

    logger.info("=== Démarrage du pipeline micro-Doppler (mode continu) ===")

    context = _build_context(cfg)

    dashboard = DashboardRadar(config=cfg, context=context)
    gen = _streaming_frame_generator(cfg, simulation=args.simulation)
    dashboard.run(gen)

    logger.info("=== Pipeline terminé ===")


if __name__ == "__main__":
    main()
