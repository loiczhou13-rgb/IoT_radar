from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .config import AppConfig
from .domain.targets import RESPIRATION
from .sdr_config import PlutoSdrConfig
from .sdr_io import PlutoSdrIO
from .tx_waveforms import WaveformSpec, generate
from .ui import create_ui, update_ui

from .processing.clutter_filter import ClutterFilterState, apply_clutter_filter
from .processing.microdoppler_stft import stft_microdoppler
from .processing.microdoppler_features import extract_features
from .processing.respiration_detector import init_state as init_detector_state, update_detector


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MicroDoppler — détection de respiration (PlutoSDR)")
    p.add_argument("--uri", default=None, help="Pluto URI, e.g. ip:192.168.2.1")
    p.add_argument("--fc", type=float, default=None, help="Carrier frequency (Hz)")
    p.add_argument("--fs", type=float, default=None, help="Sample rate (Hz)")
    p.add_argument("--rx-buffer", type=int, default=None, help="RX buffer size (samples)")
    p.add_argument("--waveform", default=None, choices=["cw_tone", "noise", "qpsk", "chirp"])
    p.add_argument("--tone-hz", type=float, default=None, help="CW tone baseband (Hz)")
    p.add_argument("--clutter", default=None, choices=["mean", "iir", "mti"], help="Clutter filter method")
    p.add_argument("--iir-alpha", type=float, default=None, help="IIR alpha for clutter removal")
    return p


def run(cli_args: list[str] | None = None) -> None:
    cfg = AppConfig()
    args = build_argparser().parse_args(cli_args)

    radio = cfg.radio
    stft_cfg = cfg.stft
    proc_cfg = cfg.processing
    det_cfg = cfg.detector

    sdr_cfg = PlutoSdrConfig(
        uri=args.uri or radio.pluto_uri,
        fc_hz=args.fc or radio.fc_hz,
        fs_hz=args.fs or radio.fs_hz,
        rx_buffer_size=args.rx_buffer or radio.rx_buffer_size,
        rx_gain_mode=radio.rx_gain_mode,
        rx_gain_db=radio.rx_gain_db,
        tx_gain_db=radio.tx_gain_db,
        tx_cyclic=True,
    )

    sdr = PlutoSdrIO(sdr_cfg)

    wf = WaveformSpec(
        kind=args.waveform or cfg.tx.waveform,
        amplitude=cfg.tx.amplitude,
        tone_hz=args.tone_hz or cfg.tx.tone_hz,
        duration_s=0.10,
    )
    tx_iq = generate(wf, fs_hz=sdr_cfg.fs_hz)
    sdr.tx(tx_iq, cyclic=True)

    clutter_method = args.clutter or cfg.clutter.method
    iir_alpha = float(args.iir_alpha) if args.iir_alpha is not None else cfg.clutter.iir_alpha

    clutter_state = ClutterFilterState()
    detector_state = init_detector_state()

    def _decimate(x: np.ndarray, decim: int) -> np.ndarray:
        if decim <= 1:
            return x
        # Simple anti-aliasing: moving-average FIR then downsample.
        k = int(decim)
        kernel = np.ones(k, dtype=np.float32) / float(k)
        xr = np.convolve(x.real.astype(np.float32), kernel, mode="same")
        xi = np.convolve(x.imag.astype(np.float32), kernel, mode="same")
        return (xr[::k] + 1j * xi[::k]).astype(np.complex64)

    fs_stft_target = float(proc_cfg.stft_fs_target_hz)
    decim = int(max(1, round(sdr_cfg.fs_hz / max(fs_stft_target, 1.0))))
    fs_stft = float(sdr_cfg.fs_hz) / float(decim)

    # Create UI after we know v-axis for microdoppler
    # Prime one RX capture to define STFT grids.
    rx0 = sdr.rx()
    y0 = apply_clutter_filter(rx0, clutter_state, method=clutter_method, iir_alpha=iir_alpha)
    y0 = _decimate(y0, decim)
    stft0 = stft_microdoppler(
        y0,
        fs_hz=fs_stft,
        fc_hz=sdr_cfg.fc_hz,
        window_size=stft_cfg.window_size,
        hop_size=stft_cfg.hop_size,
        nfft=stft_cfg.nfft,
        window=stft_cfg.window,
    )

    ui = create_ui(md_v_ms=stft0.v_ms, history_frames=stft_cfg.history_frames)

    last_update_t = time.time()

    def _frame(_i: int):
        nonlocal last_update_t, tx_iq

        rx = sdr.rx()
        y = apply_clutter_filter(rx, clutter_state, method=clutter_method, iir_alpha=iir_alpha)
        y = _decimate(y, decim)
        stft = stft_microdoppler(
            y,
            fs_hz=fs_stft,
            fc_hz=sdr_cfg.fc_hz,
            window_size=stft_cfg.window_size,
            hop_size=stft_cfg.hop_size,
            nfft=stft_cfg.nfft,
            window=stft_cfg.window,
        )

        feat = extract_features(
            v_ms=stft.v_ms,
            psd_db=stft.psd_db,
            band_v_min_ms=RESPIRATION.band_v_min_ms,
            band_v_max_ms=RESPIRATION.band_v_max_ms,
        )

        decision, _est = update_detector(
            state=detector_state,
            feature=feat,
            dt_feature_s=stft.dt_s,
            min_resp_hz=RESPIRATION.min_resp_hz,
            max_resp_hz=RESPIRATION.max_resp_hz,
            min_stable_seconds=RESPIRATION.min_stable_seconds,
            snr_db_min=det_cfg.snr_db_min,
            confidence_min=det_cfg.confidence_min,
        )

        status = "PRÉSENT" if decision.is_present else "ABSENT"
        decision_text = f"Respiration: {status} | confiance={decision.confidence:.2f} | SNR={decision.snr_db:.1f} dB"
        if decision.rate_hz > 0:
            decision_text += f" | rythme={decision.rate_bpm:.1f} bpm ({decision.rate_hz:.3f} Hz)"

        metrics_text = (
            f"STFT Δt={stft.dt_s:.4f}s  Δf={stft.df_hz:.2f}Hz  Δv={stft.dv_ms:.4g}m/s | "
            f"fe={sdr.status.applied_fs_hz:.0f}Hz  fe_STFT={fs_stft:.0f}Hz (÷{decim})  fc={sdr.status.applied_fc_hz/1e9:.3f}GHz | "
            f"filtrage_décor={clutter_method}"
        )

        # Most recent microdoppler time slice for waterfall
        md_slice = stft.psd_db[:, -1]

        update_ui(
            ui,
            fs_hz=sdr_cfg.fs_hz,
            tx_iq=tx_iq,
            rx_iq=rx,
            md_slice_db=md_slice,
            confidence=decision.confidence,
            is_present=decision.is_present,
            confidence_threshold=det_cfg.confidence_min,
            decision_text=decision_text,
            metrics_text=metrics_text,
        )

        # return artists (not using blit for simplicity)
        return (ui.line_tx, ui.line_rx, ui.img_md, ui.txt)

    anim = FuncAnimation(ui.fig, _frame, interval=stft_cfg.update_ms, blit=False, cache_frame_data=False)
    plt.show()

    sdr.stop_tx()


if __name__ == "__main__":
    run()

