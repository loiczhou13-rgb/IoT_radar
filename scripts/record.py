#!/usr/bin/env python3
"""Record labelled acquisitions of the micro-Doppler pipeline (no GUI).

Examples (from the repository root) — one 2-minute training sample, next free
index ::

    python scripts/record.py --subset train --env salle --label 1 --duration 120

Explicit sample ``AICalibration/data/train/7.*`` (label 0 = empty, 1 = presence / breathing) ::

    python scripts/record.py --subset train --env salle --label 0 --index 7 --duration 60

Five samples of 60 s with a 2-minute pause after each one (indices are then
always automatic: ``--index`` is refused) ::

    python scripts/record.py -n 5 --interval 120 --subset train --env salle --label 1 --duration 60

``--env`` must be one of the recognised environments (see ``_VALID_ENVS``).

Layout (default root: ``AICalibration/data`` in the repo) ::

    AICalibration/data/<train|test|val>/<n>.npz|.json|.iq

The metadata useful for training lives in the ``.npz``; a ``<n>.json`` file is
also written with the same information (human-readable, inventory).
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from iot_radar.acquisition.recording import (
    next_sample_index,
    write_iq_complex64,
    write_iq_stereo_wav,
)
from iot_radar.config import (
    DEFAULT_RADAR_CONFIG,
    RECORDING_DATA_DIR,
    load_config,
    radar_log_file,
    setup_logging,
)
from iot_radar.acquisition.sources import open_source
from iot_radar.pipeline import build_context, streaming_frame_generator

_VALID_ENVS: tuple[str, ...] = ("salle",)


_DEFAULT_DATA_HELP = str(RECORDING_DATA_DIR)


def build_record_argument_parser(
    *,
    description: str | None = None,
) -> argparse.ArgumentParser:
    """Build the CLI parser of one acquisition (without the repetition options)."""
    desc = description or (
        "Micro-Doppler acquisition: record the spectrogram, settings, labels, "
        "metadata and decimated IQ under "
        "AICalibration/data/<train|test|val>/<index>.*"
    )
    p = argparse.ArgumentParser(description=desc)
    p.add_argument(
        "--subset",
        required=True,
        choices=("train", "test", "val"),
        metavar="SUBSET",
        help="Sub-folder of the data root (train, test or val).",
    )
    p.add_argument(
        "--env",
        required=True,
        choices=_VALID_ENVS,
        metavar="ENV",
        help=f"Measurement environment ({', '.join(_VALID_ENVS)}).",
    )
    p.add_argument(
        "--label",
        required=True,
        type=int,
        choices=(0, 1),
        metavar="{0,1}",
        help="Class label: 0 = no target (e.g. empty room), 1 = presence / breathing.",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sample number (>= 1). Default: next free index under "
            "<data-root>/<subset>/."
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root folder of the data (default: "
            f"{_DEFAULT_DATA_HELP})."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Recording duration in seconds (default: 120 = 2 min).",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_RADAR_CONFIG),
        help="YAML configuration file.",
    )
    p.add_argument(
        "--simulation",
        action="store_true",
        help="Use the simulated source (no PlutoSDR).",
    )
    p.add_argument(
        "--no-spectrogram",
        action="store_true",
        help="Do not store the spectrogram columns (lighter file).",
    )
    p.add_argument(
        "--no-iq-file",
        action="store_true",
        help="Do not write the .iq file (raw complex64; no IQ is kept without --wav).",
    )
    p.add_argument(
        "--wav",
        action="store_true",
        help="Also write a stereo float32 .wav (I channel 0, Q channel 1).",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Log file (default: from the configuration).",
    )
    return p


def run(args: argparse.Namespace) -> Path:
    """Run one acquisition and write the ``.npz``/``.json`` (+ optional IQ/WAV).

    Returns the path to the written ``.npz``.
    """
    cfg: dict[str, Any] = load_config(args.config)
    log_path = radar_log_file(cfg, args.log_file)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"), log_path)

    logger = logging.getLogger(__name__)
    duration_s = float(args.duration)
    if duration_s <= 0:
        raise SystemExit("--duration must be > 0.")

    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else RECORDING_DATA_DIR
    )
    split_dir = (data_root / args.subset).resolve()
    split_dir.mkdir(parents=True, exist_ok=True)

    if args.index is not None:
        if args.index < 1:
            raise SystemExit("--index must be >= 1.")
        sample_index = int(args.index)
    else:
        sample_index = next_sample_index(split_dir)

    stem = str(sample_index)
    npz_path = split_dir / f"{stem}.npz"
    json_path = split_dir / f"{stem}.json"
    iq_path = split_dir / f"{stem}.iq"
    wav_path = split_dir / f"{stem}.wav"

    logger.info(
        "Recording — data_root=%s subset=%s env=%s index=%s label=%d",
        data_root,
        args.subset,
        args.env,
        stem,
        int(args.label),
    )

    context = build_context(cfg)
    f_hz = np.asarray(context["f_hz"], dtype=np.float64)
    f_s_dec_hz = float(context["f_s_dec_hz"])

    need_iq_tape = (not args.no_iq_file) or args.wav
    decimated_iq_chunks: list[np.ndarray] | None = [] if need_iq_tape else None
    gen = streaming_frame_generator(
        cfg,
        open_source(cfg, simulation=args.simulation),
        decimated_iq_chunks=decimated_iq_chunks,
    )

    t_wall: list[float] = []
    frame_numbers: list[int] = []
    spectrum_columns: list[np.ndarray] | None = [] if not args.no_spectrogram else None

    output_files = [str(npz_path), str(json_path)]
    if not args.no_iq_file:
        output_files.append(str(iq_path))
    if args.wav:
        output_files.append(str(wav_path))
    logger.info(
        "Recording for %.1f s — files: %s",
        duration_s,
        ", ".join(output_files),
    )
    if log_path:
        logger.info("Log: %s", log_path)

    t0 = time.monotonic()
    n_frames = 0
    stop_reason: str | None = None

    try:
        for frame in gen:
            elapsed = time.monotonic() - t0
            if elapsed >= duration_s:
                break

            t_wall.append(elapsed)
            frame_numbers.append(int(frame["frame_number"]))
            if spectrum_columns is not None:
                spectrum_columns.append(
                    np.asarray(frame["spectrum_column_db"], dtype=np.float64)
                )
            n_frames += 1
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        logger.warning(
            "Keyboard interrupt — saving %d frames.", n_frames,
        )
    except OSError as exc:
        en = getattr(exc, "errno", None)
        link_lost = isinstance(exc, BrokenPipeError) or en in (
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ETIMEDOUT,
            errno.ENOTCONN,
        )
        if link_lost:
            stop_reason = "pluto_link_lost"
            logger.error(
                "PlutoSDR / libiio link lost (%s). Check the USB cable or the "
                "power supply, avoid weak hubs, disable USB sleep, check the IP "
                "address (sdr.uri) and that no other program uses the Pluto. "
                "Saving %d STFT frame(s).",
                exc,
                n_frames,
            )
        else:
            raise
    finally:
        gen.close()

    has_iq_tape = bool(decimated_iq_chunks and len(decimated_iq_chunks) > 0)
    if n_frames == 0 and not has_iq_tape:
        raise SystemExit(
            "Nothing recorded. Increase the duration, check the warm-up "
            "or the PlutoSDR link."
        )
    if n_frames == 0 and has_iq_tape:
        logger.warning(
            "No STFT column (warm-up or very early interruption) — "
            "empty .npz; the partial .iq / WAV file is kept.",
        )

    label_int = int(args.label)
    recorded_wall = float(t_wall[-1]) if t_wall else 0.0

    save_kw: dict[str, Any] = {
        "t_wall_s": np.asarray(t_wall, dtype=np.float64),
        "n_trame": np.asarray(frame_numbers, dtype=np.int64),
        "f_hz": f_hz,
        "duration_requested_s": np.array(duration_s, dtype=np.float64),
        "n_frames": np.array(n_frames, dtype=np.int64),
        "label": np.array(label_int, dtype=np.int8),
        "env": np.asarray(str(args.env), dtype=str),
    }
    if spectrum_columns is not None and len(spectrum_columns) > 0:
        save_kw["spectrogram_db"] = np.stack(spectrum_columns, axis=0)

    wrote_iq = False
    wrote_wav = False
    iq_num_samples = 0

    if decimated_iq_chunks is not None and len(decimated_iq_chunks) > 0:
        iq_full = np.concatenate(decimated_iq_chunks)
        n_cap = int(round(duration_s * f_s_dec_hz))
        if n_cap > 0 and iq_full.size > n_cap:
            iq_full = iq_full[:n_cap]
        iq_num_samples = int(iq_full.size)

        if not args.no_iq_file:
            write_iq_complex64(iq_path, iq_full)
            wrote_iq = True
            logger.info(
                "Decimated IQ (.iq) — %d complex samples @ %.1f Hz — %s",
                iq_full.size,
                f_s_dec_hz,
                iq_path,
            )

        if args.wav:
            write_iq_stereo_wav(wav_path, iq_full, f_s_dec_hz)
            wrote_wav = True
            logger.info(
                "Decimated IQ WAV — %d complex samples @ %.1f Hz — %s",
                iq_full.size,
                f_s_dec_hz,
                wav_path,
            )
    else:
        if not args.no_iq_file:
            logger.warning("No IQ block collected — .iq file not created.")
        if args.wav:
            logger.warning("No IQ block — .wav file not created.")

    cfg_resolved = str(Path(args.config).resolve())
    utc_finished_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    save_kw["subset"] = np.asarray(args.subset, dtype=str)
    save_kw["sample_index"] = np.int64(sample_index)
    save_kw["config_path"] = np.asarray(cfg_resolved, dtype=str)
    save_kw["simulation"] = np.array(bool(args.simulation), dtype=np.bool_)
    save_kw["f_s_dec_hz"] = np.float64(f_s_dec_hz)
    save_kw["n_fft"] = np.int64(f_hz.size)
    save_kw["duration_recorded_wall_s"] = np.float64(recorded_wall)
    save_kw["data_root"] = np.asarray(str(data_root.resolve()), dtype=str)
    save_kw["utc_finished"] = np.asarray(utc_finished_iso, dtype=str)
    save_kw["includes_spectrogram"] = np.array(
        spectrum_columns is not None and len(spectrum_columns) > 0,
        dtype=np.bool_,
    )
    save_kw["includes_iq_file"] = np.array(wrote_iq, dtype=np.bool_)
    save_kw["includes_wav_file"] = np.array(wrote_wav, dtype=np.bool_)
    save_kw["iq_num_complex_samples"] = np.int64(iq_num_samples)
    if stop_reason is not None:
        save_kw["acquisition_stop_reason"] = np.asarray(stop_reason, dtype=str)

    np.savez_compressed(npz_path, **save_kw)

    meta_json: dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "subset": args.subset,
        "env": args.env,
        "label": label_int,
        "sample_index": sample_index,
        "sample_dir": str(split_dir.resolve()),
        "npz_file": str(npz_path.resolve()),
        "config_path": cfg_resolved,
        "simulation": bool(args.simulation),
        "duration_requested_s": duration_s,
        "duration_recorded_wall_s": recorded_wall,
        "n_frames": n_frames,
        "n_fft": int(f_hz.size),
        "f_s_dec_hz": f_s_dec_hz,
        "includes_spectrogram": bool(
            spectrum_columns is not None and len(spectrum_columns) > 0,
        ),
        "includes_iq_file": wrote_iq,
        "includes_wav_file": wrote_wav,
        "iq_num_complex_samples": iq_num_samples,
        "utc_finished": utc_finished_iso,
    }
    if wrote_iq:
        meta_json["iq_file"] = str(iq_path.resolve())
    else:
        meta_json["iq_file"] = None
        if args.no_iq_file:
            meta_json["iq_file_disabled"] = True

    if wrote_wav:
        meta_json["wav_file"] = str(wav_path.resolve())
        meta_json["wav_format"] = (
            "stereo float32; channel_0=I (real), channel_1=Q (imag)"
        )
        meta_json["wav_samples"] = iq_num_samples
    else:
        meta_json["wav_file"] = None

    if stop_reason is not None:
        meta_json["acquisition_stop_reason"] = stop_reason

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta_json, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    logger.info(
        "%d frames saved — wall time of the last frame ≈ %.2f s",
        n_frames,
        recorded_wall,
    )
    logger.info("Files: %s, %s", npz_path, json_path)
    return npz_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Acquisition options plus the repetition options ``-n`` / ``--interval``."""
    p = build_record_argument_parser()
    g = p.add_argument_group("repetition")
    g.add_argument(
        "--samples",
        "-n",
        type=int,
        default=1,
        metavar="N",
        help="Number of successive recordings (default: 1).",
    )
    g.add_argument(
        "--interval",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "Pause in seconds after one recording before the next one "
            "(default: 0)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> list[Path]:
    """CLI entry point: run ``--samples`` acquisitions (one by default)."""
    args = build_argument_parser().parse_args(argv)
    if args.samples < 1:
        raise SystemExit("-n / --samples must be >= 1.")
    if args.interval < 0:
        raise SystemExit("--interval must be >= 0.")
    if args.samples > 1 and args.index is not None:
        raise SystemExit(
            "--index is incompatible with -n > 1 (indices are assigned automatically).",
        )

    log = logging.getLogger(__name__)
    paths: list[Path] = []
    for k in range(args.samples):
        if k > 0 and args.interval > 0:
            log.info("Pause %.1f s before recording %d / %d", args.interval, k + 1, args.samples)
            time.sleep(args.interval)
        if args.samples > 1:
            log.info(
                "Recording %d / %d (subset=%s env=%s label=%s)",
                k + 1,
                args.samples,
                args.subset,
                args.env,
                args.label,
            )
        paths.append(run(args))

    if args.samples > 1:
        log.info("Done — %d .npz file(s): %s", len(paths), ", ".join(str(p) for p in paths))
    return paths


if __name__ == "__main__":
    main()
