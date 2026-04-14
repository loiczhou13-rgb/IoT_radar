"""CLI entry point for the micro-Doppler radar pipeline.

Usage
-----
Hardware mode::

    python -m MicroDopplerDetection.main --config MicroDopplerDetection/config.yaml

Simulation mode::

    python -m MicroDopplerDetection.main --config MicroDopplerDetection/config.yaml --simulation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Generator

import yaml
import numpy as np

from MicroDopplerDetection.pipeline.emission import generate_tx_buffer
from MicroDopplerDetection.pipeline.acquisition import (
    AcquisitionResult,
    acquire_pluto,
    synthesize_iq,
)
from MicroDopplerDetection.pipeline.decimation import decimate_iq
from MicroDopplerDetection.pipeline.clutter import remove_clutter
from MicroDopplerDetection.pipeline.spectrogramme import (
    SpectrogramOutput,
    compute_spectrogram,
)
from MicroDopplerDetection.pipeline.detection import (
    detect_respiration,
    snr_db_per_column,
)
from MicroDopplerDetection.utils.display import DashboardRadar

logger = logging.getLogger(__name__)


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
# Pipeline orchestration
# ------------------------------------------------------------------

def _run_acquisition(cfg: dict[str, Any], simulation: bool) -> AcquisitionResult:
    """Execute the acquisition stage (hardware or simulation)."""
    sdr = cfg["sdr"]
    emi = cfg["emission"]
    sim = cfg["simulation"]

    tx_buffer = generate_tx_buffer(
        mode=emi["mode"],
        buffer_size=sdr["buffer_size"],
        f_s=sdr["f_s"],
        f_offset=emi.get("f_offset", 0.0),
    )

    if simulation or sim.get("enable", False):
        logger.info("Mode simulation activé")
        # In simulation the IQ signal is already in baseband (no RF mixer),
        # so the breathing signature sits near 0 Hz regardless of f_offset.
        return synthesize_iq(
            f_c=sdr["f_c"],
            f_s=sdr["f_s"],
            buffer_size=sdr["buffer_size"],
            n_frames=sdr["n_frames"],
            fv=sim["fv"],
            D_mm=sim["D_mm"],
            snr_dB=sim["snr_dB"],
            f_offset=0.0,
        )

    logger.info("Mode matériel — connexion au PlutoSDR")
    return acquire_pluto(
        uri=sdr["uri"],
        f_c=sdr["f_c"],
        f_s=sdr["f_s"],
        rx_gain=sdr["rx_gain"],
        tx_gain=sdr["tx_gain"],
        buffer_size=sdr["buffer_size"],
        n_frames=sdr["n_frames"],
        tx_buffer=tx_buffer,
    )


def _run_preprocessing(
    cfg: dict[str, Any],
    acq: AcquisitionResult,
) -> tuple[np.ndarray, float]:
    """Decimate and remove clutter from the acquired IQ signal."""
    dec = cfg["decimation"]
    clu = cfg["clutter"]

    iq = acq.iq
    f_s = acq.f_s

    if dec.get("enable", True):
        iq, f_s = decimate_iq(
            iq=iq,
            f_s=f_s,
            D=dec["D"],
            f_max_utile=dec["f_max_utile"],
        )

    iq = remove_clutter(
        iq=iq,
        mode=clu["mode"],
        alpha=clu.get("alpha", 0.99),
    )

    return iq, f_s


def _run_analysis(
    cfg: dict[str, Any],
    iq: np.ndarray,
    f_s: float,
    f_c: float,
) -> SpectrogramOutput:
    """Compute the STFT spectrogram."""
    spec = cfg["spectrogramme"]
    win = cfg["windowing"]

    return compute_spectrogram(
        iq=iq,
        f_s=f_s,
        f_c=f_c,
        n_fft=spec["n_fft"],
        overlap=spec["overlap"],
        window_mode=win["mode"],
    )


def _frame_generator(
    cfg: dict[str, Any],
    iq_dec: np.ndarray,
    spect: SpectrogramOutput,
) -> Generator[dict[str, Any], None, None]:
    """Yield one frame_data dict per STFT column for the dashboard.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary.
    iq_dec : numpy.ndarray
        Decimated + clutter-filtered IQ signal.
    spect : SpectrogramOutput
        Pre-computed STFT output.

    Yields
    ------
    dict[str, Any]
        Frame data expected by :meth:`DashboardRadar.update_frame`:
        ``signal_iq_dec``, ``spectre_colonne``, ``snr_dB``, ``n_trame``,
        ``detection``.
    """
    det = cfg["detection"]
    bande_resp = tuple(det["bande_respiration"])
    bande_ref = tuple(det["bande_reference"])
    seuil = det["seuil_snr_dB"]

    snr_cols, _, _ = snr_db_per_column(
        S_db=spect.S_db,
        f_hz=spect.f_hz,
        bande_respiration=bande_resp,
        bande_reference=bande_ref,
    )

    n_time = spect.S_db.shape[1]
    samples_per_frame = max(1, len(iq_dec) // n_time)

    for col_idx in range(n_time):
        start = col_idx * samples_per_frame
        end = min(start + samples_per_frame, len(iq_dec))
        iq_chunk = iq_dec[start:end]

        snr_val = float(snr_cols[col_idx])
        alert = bool(snr_val >= seuil)

        yield {
            "signal_iq_dec": iq_chunk,
            "spectre_colonne": spect.S_db[:, col_idx],
            "snr_dB": snr_val,
            "n_trame": col_idx,
            "detection": alert,
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
    """Top-level pipeline orchestration."""
    args = _parse_args()
    cfg = _load_config(args.config)
    _setup_logging(cfg)

    logger.info("=== Démarrage du pipeline micro-Doppler ===")

    # 1. Acquisition
    acq = _run_acquisition(cfg, simulation=args.simulation)

    # 2. Preprocessing (decimation + clutter)
    iq_dec, f_s_dec = _run_preprocessing(cfg, acq)

    # 3. Spectrogram
    spect = _run_analysis(cfg, iq_dec, f_s_dec, acq.f_c)

    # 4. Global detection
    det_cfg = cfg["detection"]
    if det_cfg["mode"] == "threshold":
        result = detect_respiration(
            S_db=spect.S_db,
            f_hz=spect.f_hz,
            bande_respiration=tuple(det_cfg["bande_respiration"]),
            bande_reference=tuple(det_cfg["bande_reference"]),
            seuil_snr_dB=det_cfg["seuil_snr_dB"],
        )
        if result is not None:
            logger.info(
                "Résultat global — SNR = %.1f dB, alerte = %s",
                result.snr_db,
                result.alert,
            )

    # 5. Dashboard
    context = {
        "f_hz": spect.f_hz,
        "v_mps": spect.v_mps,
        "t_s": spect.t_s,
        "f_s_dec": f_s_dec,
        "f_c": acq.f_c,
        "seuil_snr_dB": det_cfg["seuil_snr_dB"],
    }

    dashboard = DashboardRadar(config=cfg, context=context)
    gen = _frame_generator(cfg, iq_dec, spect)
    dashboard.run(gen)

    logger.info("=== Pipeline terminé ===")


if __name__ == "__main__":
    main()
