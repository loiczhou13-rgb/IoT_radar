# PlutoFMCW/fmcw_app.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# Imports locaux
sys.path.append(os.path.abspath("../../PlutoDoppler"))
from sdr_device import PlutoDevice
from fmcw_processor import FMCWProcessor
from config import FMCWConfig
from signal_generator import SignalGenerator


class FMCWRadarApp:
    def __init__(self):
        # =========================
        # CONFIG
        # =========================
        self.fs = FMCWConfig.fs
        self.fc = FMCWConfig.fc
        self.n_samples = FMCWConfig.n_samples
        self.bw = FMCWConfig.bw
        self.tx_gain = FMCWConfig.tx_gain
        self.rx_gain = FMCWConfig.rx_gain
        self.n_history = FMCWConfig.n_history

        # Affichage / détection
        self.dist_max = 15.0
        self.max_display_frames = 120
        self.threshold_sigma = 3.0

        # Echelles fixes
        self.fft_db_ylim = (-80, 3)
        self.range_db_ylim = (-50, 100)

        # =========================
        # INITIALISATION SDR
        # =========================
        self.hw = PlutoDevice(fs=self.fs, lo=self.fc)
        self.hw.sdr.rx_buffer_size = self.n_samples * 4
        self.hw.sdr.tx_hardwaregain_chan0 = self.tx_gain
        self.hw.set_rx_gain("manual", self.rx_gain)

        # Chirp FMCW
        self.chirp_tx = SignalGenerator.generate_chirp(self.fs, self.n_samples, self.bw)
        self.hw.tx(self.chirp_tx)

        # =========================
        # INITIALISATION PROCESSEUR
        # =========================
        self.processor = FMCWProcessor(
            self.fs,
            self.bw,
            self.n_samples,
            self.chirp_tx,
            self.n_history,
        )

        # =========================
        # MÉTRIQUES (affichage bas de fenêtre)
        # =========================
        self.dist_res_m = float(self.processor.dist_res)
        self.max_range_m = float((self.n_samples // 2 - 1) * self.dist_res_m)
        self.fft_bin_hz = float(self.fs / self.n_samples)

        # =========================
        # INTERFACE
        # =========================
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=FMCWConfig.figure_size)
        # Layout : 2 FFT en haut, range profile en bas (pleine largeur)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

        self.ax_emis = self.fig.add_subplot(gs[0, 0])
        self.ax_recu = self.fig.add_subplot(gs[0, 1])
        self.ax_range = self.fig.add_subplot(gs[1, :])

        # FFT émise
        self.line_emis, = self.ax_emis.plot([], [], lw=1.2, color="#00ff88")
        self.ax_emis.set_title("FFT SIGNAL ÉMIS", color="green", fontsize=12)
        self.ax_emis.set_xlabel("Fréquence (kHz)")
        self.ax_emis.set_ylabel("Amplitude (dB)")
        self.ax_emis.grid(True, alpha=0.25, linestyle="--")

        # FFT reçue
        self.line_recu, = self.ax_recu.plot([], [], lw=1.2, color="#ffaa00")
        self.ax_recu.set_title("FFT SIGNAL REÇU", color="orange", fontsize=12)
        self.ax_recu.set_xlabel("Fréquence (kHz)")
        self.ax_recu.set_ylabel("Amplitude (dB)")
        self.ax_recu.grid(True, alpha=0.25, linestyle="--")

        # Range profile
        self.line_range, = self.ax_range.plot([], [], lw=1.5, color="cyan", label="Profil distance")
        self.scatter_range_peaks = self.ax_range.scatter([], [], color="red", s=35, marker="o", label="Pics détectés")
        self.ax_range.set_title("RANGE PROFILE / DÉTECTION", color="cyan", fontsize=12)
        self.ax_range.set_xlabel("Distance (m)")
        self.ax_range.set_ylabel("Amplitude (dB)")
        self.ax_range.grid(True, alpha=0.25, linestyle="--")
        self.ax_range.legend(loc="upper right", fontsize=9)

        # Limites initiales
        self.freq_xlim_khz = (-self.fs / 2 / 1e3, self.fs / 2 / 1e3)
        self.dist_xlim = (0, self.dist_max)

        self.ax_emis.set_xlim(*self.freq_xlim_khz)
        self.ax_emis.set_ylim(*self.fft_db_ylim)

        self.ax_recu.set_xlim(*self.freq_xlim_khz)
        self.ax_recu.set_ylim(*self.fft_db_ylim)

        self.ax_range.set_xlim(*self.dist_xlim)
        self.ax_range.set_ylim(*self.range_db_ylim)

        # Axe actif pour zoom horizontal
        self.current_ax = self.ax_recu
        self.initial_xlimits = {
            self.ax_emis: self.freq_xlim_khz,
            self.ax_recu: self.freq_xlim_khz,
            self.ax_range: self.dist_xlim,
        }

        # Indicateur simple de "mouvement" : distance(s) où la variation dépasse le seuil
        self.frame_counter = 0

        # Bandeau d'infos (bas de fenêtre)
        self.status_text = self.fig.text(
            0.01,
            0.01,
            "",
            ha="left",
            va="bottom",
            fontsize=9,
            color="#cccccc",
            family="monospace",
        )

        # Clavier
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=FMCWConfig.update_interval_ms,
            blit=False,
            cache_frame_data=False
        )

        print(f"Radar FMCW prêt. Résolution estimée : {self.processor.dist_res:.2f} m")
        print("Commandes :")
        print("  e : focus FFT émise")
        print("  r : focus FFT reçue")
        print("  p : focus range profile")
        print("  + / - : zoom horizontal")
        print("  i : reset zoom")
        print("  c : clear background + clear détections")
        print("  q : quitter")

        plt.tight_layout()
        plt.show()

    def status_line(self, n_peaks: int, threshold: float) -> str:
        return (
            f"fs={self.fs/1e6:.3f} MS/s  fc={self.fc/1e9:.3f} GHz  BW={self.bw/1e6:.3f} MHz  "
            f"ΔR≈c/(2BW)={self.dist_res_m:.3f} m  Rmax≈{self.max_range_m:.1f} m  "
            f"Δf(FFT)={self.fft_bin_hz:.1f} Hz  thresh={threshold:.3f}  peaks={n_peaks}"
        )

    @staticmethod
    def to_db_relative(x, floor_db=-200.0):
        mag = np.abs(x)
        mag = np.maximum(mag, 1e-12)
        db = 20.0 * np.log10(mag)
        db -= np.max(db)
        db = np.maximum(db, floor_db)
        return db

    def set_active_axis(self, ax):
        self.current_ax = ax
        if ax == self.ax_emis:
            print(">>> Focus sur FFT émise")
        elif ax == self.ax_recu:
            print(">>> Focus sur FFT reçue")
        elif ax == self.ax_range:
            print(">>> Focus sur range profile")

    def reset_zoom(self):
        self.current_ax.set_xlim(*self.initial_xlimits[self.current_ax])
        self.fig.canvas.draw_idle()
        print(">>> Zoom horizontal réinitialisé")

    def zoom_x_axis(self, ax, factor):
        x0, x1 = ax.get_xlim()
        cx = 0.5 * (x0 + x1)
        wx = (x1 - x0) * factor

        init_x0, init_x1 = self.initial_xlimits[ax]
        max_wx = init_x1 - init_x0
        min_wx = max_wx * 1e-4

        wx = min(wx, max_wx)
        wx = max(wx, min_wx)

        new_xlim = (cx - wx / 2, cx + wx / 2)
        ax.set_xlim(*new_xlim)
        self.fig.canvas.draw_idle()

    def clear_background(self):
        if hasattr(self.processor, "reset_background"):
            self.processor.reset_background()

        self.frame_counter = 0
        self.scatter_range_peaks.set_offsets(np.empty((0, 2)))
        self.fig.canvas.draw_idle()
        print(">>> Background reset + détections effacées")

    def update(self, frame):
        rx_data = self.hw.rx()

        try:
            dist_axis, psd_mti, variation, freq_axis, fft_emis, fft_recu = self.processor.process_frame(rx_data)
        except Exception as exc:
            print(f"[WARN] Erreur process_frame: {exc}")
            return (
                self.line_emis,
                self.line_recu,
                self.line_range,
                self.scatter_range_peaks,
                self.status_text,
            )

        # FFT émise : debug, conversion ici car fft_emis est complexe brut
        if fft_emis is not None and freq_axis is not None:
            freq_axis_khz = np.asarray(freq_axis) / 1e3
            y_emis = self.to_db_relative(fft_emis)
            self.line_emis.set_data(freq_axis_khz, y_emis)
        else:
            self.line_emis.set_data([], [])

        # FFT reçue : debug, conversion ici car fft_recu est complexe brut
        if fft_recu is not None and freq_axis is not None:
            freq_axis_khz = np.asarray(freq_axis) / 1e3
            y_recu = self.to_db_relative(fft_recu)
            self.line_recu.set_data(freq_axis_khz, y_recu)
        else:
            self.line_recu.set_data([], [])

        # Range profile + détection
        if dist_axis is None or psd_mti is None or variation is None:
            self.line_range.set_data([], [])
            self.scatter_range_peaks.set_offsets(np.empty((0, 2)))
            return (
                self.line_emis,
                self.line_recu,
                self.line_range,
                self.scatter_range_peaks,
                self.status_text,
            )

        dist_axis = np.asarray(dist_axis)
        psd_mti = np.asarray(psd_mti)
        variation = np.asarray(variation)

        valid = (dist_axis >= 0) & (dist_axis <= self.dist_max)
        dist_plot = dist_axis[valid]
        psd_plot = psd_mti[valid]
        var_plot = variation[valid]

        if dist_plot.size == 0:
            self.line_range.set_data([], [])
            self.scatter_range_peaks.set_offsets(np.empty((0, 2)))
            return (
                self.line_emis,
                self.line_recu,
                self.line_range,
                self.scatter_range_peaks,
                self.status_text,
            )

        # Important : psd_plot est déjà en dB dans le processor
        range_db = psd_plot
        self.line_range.set_data(dist_plot, range_db)

        mu = np.mean(var_plot)
        sigma = np.std(var_plot)
        threshold = mu + self.threshold_sigma * sigma
        mask = var_plot > threshold

        if np.any(mask):
            peak_x = dist_plot[mask]
            peak_y = range_db[mask]
            self.scatter_range_peaks.set_offsets(np.column_stack((peak_x, peak_y)))
        else:
            peak_x = np.array([])
            self.scatter_range_peaks.set_offsets(np.empty((0, 2)))

        # Historique des détections
        self.frame_counter += 1
        self.status_text.set_text(self.status_line(n_peaks=int(peak_x.size), threshold=float(threshold)))

        return (
            self.line_emis,
            self.line_recu,
            self.line_range,
            self.scatter_range_peaks,
            self.status_text,
        )

    def on_key(self, event):
        if event.key is None:
            return

        key = event.key.lower()

        if key == "q":
            plt.close(self.fig)
            return
        if key == "e":
            self.set_active_axis(self.ax_emis)
            return
        if key == "r":
            self.set_active_axis(self.ax_recu)
            return
        if key == "p":
            self.set_active_axis(self.ax_range)
            return
        if key == "c":
            self.clear_background()
            return
        if key == "i":
            self.reset_zoom()
            return
        if key in ["+", "equal"]:
            self.zoom_x_axis(self.current_ax, factor=0.8)
            return
        if key in ["-", "minus"]:
            self.zoom_x_axis(self.current_ax, factor=1.25)
            return


if __name__ == "__main__":
    app = FMCWRadarApp()