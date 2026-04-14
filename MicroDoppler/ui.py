from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class UiHandles:
    fig: plt.Figure
    ax_tx: plt.Axes
    ax_rx: plt.Axes
    ax_md: plt.Axes

    line_tx: any
    line_rx: any
    img_md: any
    txt: any

    initialized: bool = False
    rx_ylim_fixed: tuple[float, float] = (-140.0, -10.0)
    tx_ylim_fixed: tuple[float, float] = (-160.0, 20.0)
    xlim_tx: tuple[float, float] | None = None
    xlim_rx: tuple[float, float] | None = None


def _psd_db(x: np.ndarray, fs_hz: float, nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.complex64)
    if x.size == 0:
        f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs_hz)).astype(np.float32)
        return f, np.full_like(f, -140.0)
    w = np.hanning(min(x.size, nfft)).astype(np.float32)
    xw = x[: w.size] * w
    X = np.fft.fft(xw, n=nfft)
    X = np.fft.fftshift(X)
    mag = np.abs(X).astype(np.float32)
    db = 20.0 * np.log10(mag / (w.size + 1e-12) + 1e-12)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs_hz)).astype(np.float32)
    return f, db


def create_ui(
    *,
    md_v_ms: np.ndarray,
    history_frames: int,
    title: str = "MicroDoppler — Détection de respiration (interface minimale)",
    rx_ylim: tuple[float, float] = (-140.0, -10.0),
) -> UiHandles:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title(title)

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.25, wspace=0.15)
    ax_tx = fig.add_subplot(gs[0, 0])
    ax_rx = fig.add_subplot(gs[0, 1])
    ax_md = fig.add_subplot(gs[1, :])

    ax_tx.set_title("Spectre émission (bande de base)")
    ax_rx.set_title("Spectre réception (bande de base)")
    ax_md.set_title("Micro‑Doppler — spectrogramme (vitesse)")

    ax_tx.set_xlabel("Fréquence (Hz)")
    ax_rx.set_xlabel("Fréquence (Hz)")
    ax_md.set_xlabel("Vitesse (m/s)")
    ax_md.set_ylabel("Temps (trames)")

    line_tx, = ax_tx.plot([], [], lw=1.0, color="#00d4ff")
    line_rx, = ax_rx.plot([], [], lw=1.0, color="#ffb000")

    md_buf = np.full((history_frames, md_v_ms.size), -120.0, dtype=np.float32)
    img_md = ax_md.imshow(
        md_buf,
        aspect="auto",
        origin="lower",
        extent=[float(md_v_ms[0]), float(md_v_ms[-1]), 0, history_frames],
        cmap="magma",
        vmin=-120,
        vmax=-30,
        interpolation="nearest",
    )
    fig.colorbar(img_md, ax=ax_md, pad=0.01, label="dB")

    txt = fig.text(
        0.01,
        0.985,
        "",
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        color="#00ff41",
    )

    ui = UiHandles(
        fig=fig,
        ax_tx=ax_tx,
        ax_rx=ax_rx,
        ax_md=ax_md,
        line_tx=line_tx,
        line_rx=line_rx,
        img_md=img_md,
        txt=txt,
        rx_ylim_fixed=(float(rx_ylim[0]), float(rx_ylim[1])),
    )

    def _zoom_at(ax: plt.Axes, xdata: float, ydata: float, scale: float) -> None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if not (np.isfinite(xdata) and np.isfinite(ydata)):
            xdata = 0.5 * (x0 + x1)
            ydata = 0.5 * (y0 + y1)

        new_w = (x1 - x0) * scale
        new_h = (y1 - y0) * scale
        ax.set_xlim(xdata - new_w / 2.0, xdata + new_w / 2.0)
        ax.set_ylim(ydata - new_h / 2.0, ydata + new_h / 2.0)

    def _on_scroll(event):
        ax = event.inaxes
        if ax is None:
            return
        # molette vers le haut => zoom in
        scale = 0.85 if event.button == "up" else 1.0 / 0.85
        _zoom_at(ax, event.xdata, event.ydata, scale)
        ui.fig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "r":
            # reset zoom to initial limits
            if ui.xlim_tx is not None:
                ui.ax_tx.set_xlim(*ui.xlim_tx)
            if ui.xlim_rx is not None:
                ui.ax_rx.set_xlim(*ui.xlim_rx)
            ui.ax_tx.set_ylim(*ui.tx_ylim_fixed)
            ui.ax_rx.set_ylim(*ui.rx_ylim_fixed)
            ui.fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", _on_scroll)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    return ui


def update_ui(
    ui: UiHandles,
    *,
    fs_hz: float,
    tx_iq: np.ndarray,
    rx_iq: np.ndarray,
    md_slice_db: np.ndarray,
    decision_text: str,
    metrics_text: str,
) -> None:
    f_tx, p_tx = _psd_db(tx_iq, fs_hz)
    f_rx, p_rx = _psd_db(rx_iq, fs_hz)

    ui.line_tx.set_data(f_tx, p_tx)
    ui.line_rx.set_data(f_rx, p_rx)

    # Initialize axes once (matplotlib default limits are not (0, 1) in all backends).
    if not ui.initialized:
        ui.ax_tx.set_xlim(float(f_tx[0]), float(f_tx[-1]))
        ui.ax_rx.set_xlim(float(f_rx[0]), float(f_rx[-1]))
        # Use a wide default range; we refine after a few frames.
        ui.ax_tx.set_ylim(*ui.tx_ylim_fixed)
        ui.ax_rx.set_ylim(*ui.rx_ylim_fixed)
        ui.xlim_tx = (float(f_tx[0]), float(f_tx[-1]))
        ui.xlim_rx = (float(f_rx[0]), float(f_rx[-1]))
        ui.initialized = True
    else:
        # TX: adaptatif pour rester visible.
        p_lo = float(np.nanpercentile(p_tx, 5))
        p_hi = float(np.nanpercentile(p_tx, 99))
        if np.isfinite(p_lo) and np.isfinite(p_hi) and (p_hi - p_lo) > 1:
            ui.ax_tx.set_ylim(p_lo - 10, p_hi + 5)
        # RX: axe Y fixe (demande utilisateur) => ne pas auto-scale.

    # update microdoppler buffer (roll)
    buf = ui.img_md.get_array()
    buf = np.roll(buf, -1, axis=0)
    buf[-1, :] = md_slice_db.astype(np.float32)
    ui.img_md.set_array(buf)

    ui.txt.set_text(f"{decision_text}\n{metrics_text}")

