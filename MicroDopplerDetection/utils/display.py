"""Real-time matplotlib dashboard for the micro-Doppler radar pipeline."""

from __future__ import annotations

import logging
from typing import Any, Generator

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)


class DashboardRadar:
    """Six-panel live dashboard for micro-Doppler monitoring.

    Parameters
    ----------
    config : dict
        Full configuration dictionary (parsed ``config.yaml``).
    context : dict
        Pipeline context built by ``main.py``, containing at least:

        * ``f_hz``   — Doppler frequency axis (numpy.ndarray).
        * ``v_mps``  — radial velocity axis (numpy.ndarray).
        * ``t_s``    — slow-time axis (numpy.ndarray).
        * ``f_s_dec`` — decimated sampling rate (float).
        * ``f_c``    — carrier frequency (float).
        * ``seuil_snr_dB`` — detection threshold (float).

    Notes
    -----
    The dashboard comprises six panels laid out in a 3×2 grid:

    1. **IQ temporal** — real part of the latest decimated IQ frame.
    2. **Waterfall** — scrolling time–frequency spectrogram (dB colour map).
    3. **Spectral profile** — average Doppler spectrum (last N columns).
    4. **Current spectrum column** — single STFT column from the latest frame.
    5. **SNR history** — rolling SNR curve with threshold line.
    6. **Presence probability** — sigmoid-transformed SNR over time.

    The dashboard is driven by ``matplotlib.animation.FuncAnimation``
    which calls :meth:`update_frame` for each new data dict yielded by the
    pipeline generator.
    """

    def __init__(self, config: dict, context: dict) -> None:
        aff = config["affichage"]
        det = config["detection"]

        self._colormap: str = aff["colormap"]
        self._dyn_dB: float = aff["dynamique_dB"]
        self._ylim: list[float] = aff["ylim"]
        self._waterfall_s: float = aff["fenetre_waterfall_s"]
        self._N_profil: int = aff["N_profil_moyen"]
        self._N_hist: int = aff["N_historique"]
        self._sigmoid_scale: float = aff["echelle_sigmoid_dB"]
        self._plein_ecran: bool = aff["plein_ecran"]
        self._seuil_snr: float = det["seuil_snr_dB"]

        self._f_hz: np.ndarray = context["f_hz"]
        self._v_mps: np.ndarray = context["v_mps"]

        self._snr_history: list[float] = []
        self._prob_history: list[float] = []
        self._waterfall_buf: list[np.ndarray] = []
        self._frame_count: int = 0

        self._fig, self._axes = plt.subplots(
            3, 2, figsize=(14, 9), constrained_layout=True,
        )
        self._fig.suptitle(
            "Radar Micro-Doppler — Détection de survivants",
            fontsize=13,
            fontweight="bold",
        )

        self._init_panels()

        if self._plein_ecran:
            mng = plt.get_current_fig_manager()
            if mng is not None:
                try:
                    mng.full_screen_toggle()
                except Exception:
                    pass

        logger.info("Dashboard initialisé — 6 panneaux, colormap='%s'", self._colormap)

    # ------------------------------------------------------------------
    # Panel initialisation
    # ------------------------------------------------------------------

    def _init_panels(self) -> None:
        """Set up axes, labels, and placeholder artists for all six panels."""
        ax_iq, ax_waterfall = self._axes[0]
        ax_profil, ax_colonne = self._axes[1]
        ax_snr, ax_prob = self._axes[2]

        # Panel 1 — IQ temporal
        ax_iq.set_title("Signal IQ décimé (Re)")
        ax_iq.set_xlabel("Échantillon")
        ax_iq.set_ylabel("Amplitude")
        (self._line_iq,) = ax_iq.plot([], [], linewidth=0.5, color="tab:blue")

        # Panel 2 — Waterfall
        ax_waterfall.set_title("Waterfall (spectrogramme)")
        ax_waterfall.set_xlabel("Trame")
        ax_waterfall.set_ylabel("Vitesse (m/s)")
        self._img_waterfall = ax_waterfall.imshow(
            np.zeros((len(self._v_mps), 1)),
            aspect="auto",
            origin="lower",
            cmap=self._colormap,
            extent=[0, 1, self._ylim[0], self._ylim[1]],
            vmin=-self._dyn_dB,
            vmax=0,
        )
        self._fig.colorbar(self._img_waterfall, ax=ax_waterfall, label="dB")

        # Panel 3 — Average spectral profile
        ax_profil.set_title("Profil spectral moyen")
        ax_profil.set_xlabel("Fréquence Doppler (Hz)")
        ax_profil.set_ylabel("Puissance (dB)")
        (self._line_profil,) = ax_profil.plot(
            self._f_hz, np.zeros(len(self._f_hz)), linewidth=0.8, color="tab:orange",
        )

        # Panel 4 — Latest spectrum column
        ax_colonne.set_title("Spectre instantané")
        ax_colonne.set_xlabel("Fréquence Doppler (Hz)")
        ax_colonne.set_ylabel("Puissance (dB)")
        (self._line_colonne,) = ax_colonne.plot(
            self._f_hz, np.zeros(len(self._f_hz)), linewidth=0.8, color="tab:green",
        )

        # Panel 5 — SNR history
        ax_snr.set_title("Historique SNR")
        ax_snr.set_xlabel("Trame")
        ax_snr.set_ylabel("SNR (dB)")
        (self._line_snr,) = ax_snr.plot([], [], linewidth=1, color="tab:red")
        ax_snr.axhline(
            self._seuil_snr, color="grey", linestyle="--", linewidth=0.8, label="Seuil",
        )
        ax_snr.legend(loc="upper left", fontsize=8)

        # Panel 6 — Presence probability
        ax_prob.set_title("Probabilité de présence")
        ax_prob.set_xlabel("Trame")
        ax_prob.set_ylabel("P(présence)")
        ax_prob.set_ylim(-0.05, 1.05)
        (self._line_prob,) = ax_prob.plot([], [], linewidth=1, color="tab:purple")
        ax_prob.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)

    # ------------------------------------------------------------------
    # Frame update (called by FuncAnimation)
    # ------------------------------------------------------------------

    def update_frame(self, frame_data: dict[str, Any]) -> tuple:
        """Refresh all dashboard panels with new pipeline data.

        Parameters
        ----------
        frame_data : dict
            Dictionary produced by the pipeline generator, containing:

            * ``signal_iq_dec`` — latest decimated IQ frame (complex ndarray).
            * ``spectre_colonne`` — single dB column, shape ``(n_fft,)``.
            * ``snr_dB`` — scalar SNR for this frame.
            * ``n_trame`` — frame counter (int).
            * ``detection`` — boolean alert flag.

        Returns
        -------
        tuple
            Matplotlib artists that were modified (for blitting).
        """
        self._frame_count += 1

        iq_frame = frame_data["signal_iq_dec"]
        col_db = frame_data["spectre_colonne"]
        snr = frame_data["snr_dB"]
        n_trame = frame_data["n_trame"]
        detected = frame_data["detection"]

        # Panel 1 — IQ
        re = np.real(iq_frame)
        self._line_iq.set_data(np.arange(len(re)), re)
        ax_iq = self._axes[0, 0]
        ax_iq.set_xlim(0, len(re))
        margin = max(np.max(np.abs(re)) * 1.1, 1e-6)
        ax_iq.set_ylim(-margin, margin)

        # Panel 2 — Waterfall
        self._waterfall_buf.append(col_db.copy())
        waterfall = np.column_stack(self._waterfall_buf)
        v_max = float(np.max(waterfall))
        self._img_waterfall.set_data(waterfall)
        self._img_waterfall.set_extent(
            [0, waterfall.shape[1], self._ylim[0], self._ylim[1]],
        )
        self._img_waterfall.set_clim(vmin=v_max - self._dyn_dB, vmax=v_max)
        self._axes[0, 1].set_xlim(0, waterfall.shape[1])

        # Panel 3 — Average profile (last N columns)
        recent = self._waterfall_buf[-self._N_profil :]
        avg_profile = np.mean(np.column_stack(recent), axis=1)
        self._line_profil.set_ydata(avg_profile)
        ax_profil = self._axes[1, 0]
        ax_profil.set_xlim(self._f_hz[0], self._f_hz[-1])
        _auto_ylim(ax_profil, avg_profile)

        # Panel 4 — Instantaneous column
        self._line_colonne.set_ydata(col_db)
        ax_colonne = self._axes[1, 1]
        ax_colonne.set_xlim(self._f_hz[0], self._f_hz[-1])
        _auto_ylim(ax_colonne, col_db)

        # Panel 5 — SNR history
        self._snr_history.append(snr)
        if len(self._snr_history) > self._N_hist:
            self._snr_history = self._snr_history[-self._N_hist :]
        x_snr = np.arange(len(self._snr_history))
        self._line_snr.set_data(x_snr, self._snr_history)
        ax_snr = self._axes[2, 0]
        ax_snr.set_xlim(0, max(len(self._snr_history), 1))
        _auto_ylim(ax_snr, np.array(self._snr_history), margin_ratio=0.2)

        # Panel 6 — Presence probability (sigmoid of SNR)
        prob = _sigmoid(snr, self._seuil_snr, self._sigmoid_scale)
        self._prob_history.append(prob)
        if len(self._prob_history) > self._N_hist:
            self._prob_history = self._prob_history[-self._N_hist :]
        x_prob = np.arange(len(self._prob_history))
        self._line_prob.set_data(x_prob, self._prob_history)
        self._axes[2, 1].set_xlim(0, max(len(self._prob_history), 1))

        # Title colour feedback
        colour = "green" if detected else "red"
        status = "RESPIRATION DÉTECTÉE" if detected else "Aucune détection"
        self._fig.suptitle(
            f"Radar Micro-Doppler — {status}  |  Trame {n_trame}  |  "
            f"SNR = {snr:.1f} dB",
            fontsize=13,
            fontweight="bold",
            color=colour,
        )

        return (
            self._line_iq,
            self._img_waterfall,
            self._line_profil,
            self._line_colonne,
            self._line_snr,
            self._line_prob,
        )

    # ------------------------------------------------------------------
    # Animation loop
    # ------------------------------------------------------------------

    def run(self, generator: Generator[dict[str, Any], None, None]) -> None:
        """Start the live animation driven by a frame generator.

        Parameters
        ----------
        generator : Generator[dict[str, Any], None, None]
            Yields one ``frame_data`` dict per pipeline iteration.

        Notes
        -----
        ``FuncAnimation`` pulls one frame per call from the generator.
        The dashboard remains responsive because matplotlib's event loop
        handles redraw and user interaction between frames.
        """
        logger.info("Lancement du dashboard temps réel")

        self._anim = FuncAnimation(
            self._fig,
            self.update_frame,
            frames=generator,
            interval=50,
            blit=False,
            cache_frame_data=False,
            repeat=False,
        )
        plt.show()


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------

def _sigmoid(x: float, center: float, scale: float) -> float:
    """Logistic sigmoid: 1 / (1 + exp(-(x - center) / scale))."""
    z = -(x - center) / scale
    z = max(min(z, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(z))


def _auto_ylim(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    margin_ratio: float = 0.1,
) -> None:
    """Set y-axis limits with a small margin around the data range."""
    if len(data) == 0:
        return
    lo, hi = float(np.min(data)), float(np.max(data))
    span = hi - lo if hi > lo else 1.0
    margin = span * margin_ratio
    ax.set_ylim(lo - margin, hi + margin)
