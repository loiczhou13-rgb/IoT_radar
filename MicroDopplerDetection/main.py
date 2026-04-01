#!/usr/bin/env python3
"""
Entry point: load YAML, run emission → acquisition → processing → display.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

# Ensure package imports resolve when launched as `python main.py`
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.acquisition import AcquisitionResult, acquire_pluto, synthesize_iq
from pipeline.clutter import ClutterMode, suppress_clutter
from pipeline.decimation import decimate_iq
from pipeline.detection import DetectionMode, detect_respiration
from pipeline.emission import EmissionMode, generate_tx_waveform
from pipeline.spectrogramme import SpectrogramOutput, compute_spectrogram
from pipeline.windowing import WindowMode
from utils.display import DisplayConfig, show_waterfall

logger = logging.getLogger("radar")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError("config root must be a mapping")
    return dict(data)


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def run_pipeline(cfg: Mapping[str, Any], simulation_override: bool) -> None:
    sdr = cfg["sdr"]
    emission = cfg["emission"]
    decim = cfg["decimation"]
    clutter = cfg["clutter"]
    windowing = cfg["windowing"]
    spec_cfg = cfg["spectrogramme"]
    det = cfg["detection"]
    aff = cfg["affichage"]
    sim = cfg["simulation"]

    use_sim = bool(sim.get("enable", False)) or simulation_override
    f_c = float(sdr["f_c"])
    f_s = float(sdr["f_s"])
    buffer_size = int(sdr["buffer_size"])
    n_frames = int(sdr["n_frames"])

    if use_sim:
        logger.info("Running in SIMULATION mode (no PlutoSDR).")
        n_total = buffer_size * n_frames
        rng = np.random.default_rng(42)
        iq = synthesize_iq(
            f_s=f_s,
            f_c=f_c,
            n_samples=n_total,
            fv_hz=float(sim["fv"]),
            displacement_m=float(sim["D_mm"]) / 1000.0,
            snr_db=float(sim["snr_dB"]),
            rng=rng,
        )
        acq = AcquisitionResult(iq=iq, duration_s=float(iq.shape[0]) / f_s, f_s=f_s, f_c=f_c)
    else:
        mode: EmissionMode = str(emission["mode"])  # type: ignore[assignment]
        f_off = float(emission.get("f_offset", 0.0))
        tx_buf = generate_tx_waveform(
            mode,
            n_samples=buffer_size,
            f_s=f_s,
            f_offset_hz=f_off,
        )
        acq = acquire_pluto(
            uri=str(sdr["uri"]),
            f_c=f_c,
            f_s=f_s,
            rx_gain_db=float(sdr["rx_gain"]),
            tx_gain_db=float(sdr["tx_gain"]),
            buffer_size=buffer_size,
            n_frames=n_frames,
            tx_iq=tx_buf,
        )

    iq_dec, f_s_dec = decimate_iq(
        acq.iq,
        f_s=acq.f_s,
        factor=int(decim["D"]),
        f_max_utile_hz=float(decim["f_max_utile"]),
        enabled=bool(decim.get("enable", True)),
    )

    clutter_mode: ClutterMode = str(clutter["mode"])  # type: ignore[assignment]
    iq_filt = suppress_clutter(
        iq_dec,
        mode=clutter_mode,
        alpha=float(clutter.get("alpha", 0.99)),
    )

    win_mode: WindowMode = str(windowing["mode"])  # type: ignore[assignment]
    spec: SpectrogramOutput = compute_spectrogram(
        iq_filt,
        f_s=f_s_dec,
        f_c=f_c,
        n_fft=int(spec_cfg["n_fft"]),
        overlap_ratio=float(spec_cfg["overlap"]),
        window_mode=win_mode,
    )

    det_mode: DetectionMode = str(det["mode"])  # type: ignore[assignment]
    band_r = (float(det["bande_respiration"][0]), float(det["bande_respiration"][1]))
    band_n = (float(det["bande_reference"][0]), float(det["bande_reference"][1]))
    det_res = detect_respiration(
        spec.Z,
        spec.f_hz,
        band_respiration_hz=band_r,
        band_reference_hz=band_n,
        seuil_snr_db=float(det["seuil_snr_dB"]),
        mode=det_mode,
        df_hz=spec.df_hz,
    )
    if det_res is not None and det_res.alert:
        logger.warning("DETECTION ALERT: SNR %.2f dB exceeds threshold.", det_res.snr_db)

    ylim = aff["ylim"]
    disp = DisplayConfig(
        dynamique_dB=float(aff["dynamique_dB"]),
        ylim=(float(ylim[0]), float(ylim[1])),
        colormap=str(aff["colormap"]),
        f_c=f_c,
        f_s=f_s_dec,
        df_hz=spec.df_hz,
        dv_mps=spec.dv_mps,
    )
    show_waterfall(spec, disp, title="Micro-Doppler spectrogram (PlutoSDR pipeline)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Micro-Doppler radar pipeline (PlutoSDR).")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "config.yaml",
        help="Path to YAML configuration (default: ./config.yaml next to main.py).",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Force simulation mode even if simulation.enable is false in YAML.",
    )
    args = parser.parse_args(argv)

    cfg_path: Path = args.config
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = _load_yaml(cfg_path)
    log_cfg = cfg.get("logging", {})
    _setup_logging(str(log_cfg.get("level", "INFO")))

    try:
        run_pipeline(cfg, simulation_override=bool(args.simulation))
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
