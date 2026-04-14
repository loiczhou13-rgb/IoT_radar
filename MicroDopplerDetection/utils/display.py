"""Real-time matplotlib dashboard for the micro-Doppler radar pipeline."""

from __future__ import annotations

import logging
from typing import Any, Generator

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


class DashboardRadar:
    """Three-panel live dashboard with an info box for micro-Doppler monitoring.

    Layout (using GridSpec)::

        Row 0 : TX spectrum (full width)
        Row 1 : RX spectrum (full width)
        Row 2 : Probability curve (left 2/3) | Info box (right 1/3)
    """

    def __init__(self, config: dict, context: dict) -> None:
        aff = config["affichage"]
        det = config["detection"]

        self._N_hist: int = aff["N_historique"]
        self._sigmoid_scale: float = aff["echelle_sigmoid_dB"]
        self._plein_ecran: bool = aff["plein_ecran"]
        self._seuil_snr: float = det["seuil_snr_dB"]
        self._seuil_proba: float = aff.get("seuil_proba", 0.6)

        self._f_hz: np.ndarray = context["f_hz"]
        self._f_hz_tx: np.ndarray = context["f_hz_tx"]
        self._spectre_tx_db: np.ndarray = context["spectre_tx_db"]
        self._R_min_m: float = context.get("R_min_m", 0.0)
        self._R_max_m: float = context.get("R_max_m", 0.0)
        self._df_hz: float = context.get("df_hz", 0.0)
        self._dv_mps: float = context.get("dv_mps", 0.0)
        self._n_fft: int = context.get("n_fft", 0)
        self._clutter_mode: str = context.get("clutter_mode", "?")
        self._bande_resp: list = context.get("bande_resp", [0.1, 1.0])
        self._B_eff_hz: float = context.get("B_eff_hz", 0.0)

        self._prob_history: list[float] = []
        self._frame_count: int = 0
        self._last_snr: float = 0.0
        self._last_prob: float = 0.0

        self._fig = plt.figure(figsize=(14, 9), constrained_layout=True)
        gs = GridSpec(3, 5, figure=self._fig)

        self._ax_tx = self._fig.add_subplot(gs[0, :])
        self._ax_rx = self._fig.add_subplot(gs[1, :])
        self._ax_prob = self._fig.add_subplot(gs[2, :3])
        self._ax_info = self._fig.add_subplot(gs[2, 3:])

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

        logger.info("Dashboard initialisé — 3 panneaux + encadré info")

    # ------------------------------------------------------------------
    # Panel initialisation
    # ------------------------------------------------------------------

    def _init_panels(self) -> None:
        ax_tx = self._ax_tx
        ax_rx = self._ax_rx
        ax_prob = self._ax_prob
        ax_info = self._ax_info

        # Panel 1 — TX spectrum (static)
        ax_tx.set_title("Signal émis (domaine fréquentiel)")
        ax_tx.set_xlabel("Fréquence (Hz)")
        ax_tx.set_ylabel("Puissance (dB)")
        ax_tx.plot(
            self._f_hz_tx,
            self._spectre_tx_db,
            linewidth=0.8,
            color="tab:blue",
        )
        _auto_ylim(ax_tx, self._spectre_tx_db)

        # Panel 2 — RX spectrum (live)
        ax_rx.set_title("Signal reçu (domaine fréquentiel)")
        ax_rx.set_xlabel("Fréquence Doppler (Hz)")
        ax_rx.set_ylabel("Puissance (dB)")
        (self._line_rx,) = ax_rx.plot(
            self._f_hz,
            np.zeros(len(self._f_hz)),
            linewidth=0.8,
            color="tab:orange",
        )

        # Panel 3a — Presence probability
        ax_prob.set_title("Probabilité de présence")
        ax_prob.set_xlabel("Trame")
        ax_prob.set_ylabel("P(présence)")
        ax_prob.set_ylim(-0.05, 1.05)
        (self._line_prob,) = ax_prob.plot(
            [], [], linewidth=1.2, color="tab:purple",
        )
        ax_prob.axhline(
            self._seuil_proba,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            label=f"Seuil ({self._seuil_proba})",
        )
        ax_prob.legend(loc="upper left", fontsize=8)

        # Panel 3b — Info box (static structure, updated text)
        ax_info.set_axis_off()
        self._info_text = ax_info.text(
            0.05, 0.95, "",
            transform=ax_info.transAxes,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#f0f0f0",
                edgecolor="#888888",
                linewidth=1.2,
            ),
        )
        self._update_info_box()

    # ------------------------------------------------------------------
    # Info box
    # ------------------------------------------------------------------

    def _update_info_box(self) -> None:
        lines = [
            "╔══════════════════════╗",
            "║   PARAMÈTRES RADAR   ║",
            "╚══════════════════════╝",
            "",
            f"  δf     = {self._df_hz:.3f} Hz",
            f"  δv     = {self._dv_mps * 100:.2f} cm/s",
            f"  N_FFT  = {self._n_fft}",
            f"  B_eff  = {self._B_eff_hz:.1f} Hz",
            f"  Clutter: {self._clutter_mode}",
            f"  Bande  : {self._bande_resp[0]}–{self._bande_resp[1]} Hz",
            "",
            f"  Portée : {self._R_min_m:.1f}–{self._R_max_m:.1f} m",
            "",
            "─────── LIVE ───────",
            f"  SNR    = {self._last_snr:+.1f} dB",
            f"  P(vie) = {self._last_prob:.2f}",
        ]
        self._info_text.set_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Frame update (called by FuncAnimation)
    # ------------------------------------------------------------------

    def update_frame(self, frame_data: dict[str, Any]) -> tuple:
        self._frame_count += 1

        col_db = frame_data["spectre_colonne"]
        snr = frame_data["snr_dB"]
        n_trame = frame_data["n_trame"]
        detected = frame_data["detection"]

        # Panel 2 — RX spectrum
        self._line_rx.set_ydata(col_db)
        ax_rx = self._ax_rx
        ax_rx.set_xlim(self._f_hz[0], self._f_hz[-1])
        _auto_ylim(ax_rx, col_db)

        # Panel 3a — Presence probability
        prob = _sigmoid(snr, self._seuil_snr, self._sigmoid_scale)
        self._prob_history.append(prob)
        if len(self._prob_history) > self._N_hist:
            self._prob_history = self._prob_history[-self._N_hist:]
        x_prob = np.arange(len(self._prob_history))
        self._line_prob.set_data(x_prob, self._prob_history)
        self._ax_prob.set_xlim(0, max(len(self._prob_history), 1))

        # Panel 3b — Info box live values
        self._last_snr = snr
        self._last_prob = prob
        self._update_info_box()

        # Title feedback
        colour = "green" if detected else "red"
        status = "RESPIRATION DÉTECTÉE" if detected else "Aucune détection"
        self._fig.suptitle(
            f"Radar Micro-Doppler — {status}  |  Trame {n_trame}",
            fontsize=13,
            fontweight="bold",
            color=colour,
        )

        return (self._line_rx, self._line_prob, self._info_text)

    # ------------------------------------------------------------------
    # Animation loop
    # ------------------------------------------------------------------

    def run(self, generator: Generator[dict[str, Any], None, None]) -> None:
        """Start the live animation driven by a frame generator."""
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
