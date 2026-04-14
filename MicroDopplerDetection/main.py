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
        logger.info("Mode simulation continu activé")
        return stream_simulation(
            f_c=sdr["f_c"],
            f_s=sdr["f_s"],
            buffer_size=sdr["buffer_size"],
            fv=sim["fv"],
            D_mm=sim["D_mm"],
            snr_dB=sim["snr_dB"],
            f_offset=0.0,
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

    bande_resp = tuple(det_cfg["bande_respiration"])
    bande_ref = tuple(det_cfg["bande_reference"])
    seuil = det_cfg["seuil_snr_dB"]

    ring = deque(maxlen=n_fft)
    iq_stream = _build_iq_stream(cfg, simulation)

    logger.info(
        "Pipeline streaming — n_fft=%d, hop=%d, f_s_dec=%.1f Hz",
        n_fft,
        hop,
        f_s_dec,
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
            segment = np.array(ring, dtype=np.complex64)

            col = compute_single_column(segment, f_s_dec, f_c, window)

            snr_db = detect_snr_column(
                col_db=col.col_db,
                f_hz=col.f_hz,
                bande_respiration=bande_resp,
                bande_reference=bande_ref,
            )
            alert = bool(snr_db >= seuil)

            yield {
                "signal_iq_dec": segment,
                "spectre_colonne": col.col_db,
                "snr_dB": snr_db,
                "n_trame": frame_counter,
                "detection": alert,
            }
            frame_counter += 1


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

    return {
        "f_hz": f_hz,
        "f_s_dec": f_s_dec,
        "f_c": f_c,
        "seuil_snr_dB": det_cfg["seuil_snr_dB"],
        "spectre_tx_db": spectre_tx_db,
        "f_hz_tx": f_hz_tx,
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
