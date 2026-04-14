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
    """Three-panel live dashboard for micro-Doppler monitoring.

    Parameters
    ----------
    config : dict
        Full configuration dictionary (parsed ``config.yaml``).
    context : dict
        Pipeline context built by ``main.py``, containing at least:

        * ``f_hz``       — Doppler frequency axis (numpy.ndarray).
        * ``f_s_dec``    — decimated sampling rate (float).
        * ``f_c``        — carrier frequency (float).
        * ``seuil_snr_dB`` — SNR detection threshold (float).
        * ``spectre_tx_db`` — TX spectrum in dB (numpy.ndarray).
        * ``f_hz_tx``    — TX frequency axis (numpy.ndarray).

    Notes
    -----
    The dashboard has three vertically stacked panels:

    1. **Emitted signal (TX)** — frequency-domain view of the transmit
       waveform.  This is static (computed once at startup).
    2. **Received signal (RX)** — live frequency-domain view of the latest
       STFT column (Doppler spectrum after clutter removal).
    3. **Presence probability** — sigmoid-transformed SNR over time,
       with a configurable decision threshold line at 0.6 (or as set in
       ``affichage.seuil_proba``).

    The dashboard is driven by ``matplotlib.animation.FuncAnimation``
    which calls :meth:`update_frame` for each new data dict yielded by
    the pipeline generator.
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

        self._prob_history: list[float] = []
        self._frame_count: int = 0

        self._fig, self._axes = plt.subplots(
            3, 1, figsize=(12, 9), constrained_layout=True,
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

        logger.info("Dashboard initialisé — 3 panneaux")

    # ------------------------------------------------------------------
    # Panel initialisation
    # ------------------------------------------------------------------

    def _init_panels(self) -> None:
        """Set up axes, labels, and placeholder artists for all three panels."""
        ax_tx = self._axes[0]
        ax_rx = self._axes[1]
        ax_prob = self._axes[2]

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

        # Panel 3 — Presence probability
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

    # ------------------------------------------------------------------
    # Frame update (called by FuncAnimation)
    # ------------------------------------------------------------------

    def update_frame(self, frame_data: dict[str, Any]) -> tuple:
        """Refresh the RX spectrum and probability panels.

        Parameters
        ----------
        frame_data : dict
            Dictionary produced by the pipeline generator, containing:

            * ``spectre_colonne`` — RX dB spectrum, shape ``(n_fft,)``.
            * ``snr_dB`` — scalar SNR for this frame.
            * ``n_trame`` — frame counter (int).
            * ``detection`` — boolean alert flag.

        Returns
        -------
        tuple
            Matplotlib artists that were modified.
        """
        self._frame_count += 1

        col_db = frame_data["spectre_colonne"]
        snr = frame_data["snr_dB"]
        n_trame = frame_data["n_trame"]
        detected = frame_data["detection"]

        # Panel 2 — RX spectrum
        self._line_rx.set_ydata(col_db)
        ax_rx = self._axes[1]
        ax_rx.set_xlim(self._f_hz[0], self._f_hz[-1])
        _auto_ylim(ax_rx, col_db)

        # Panel 3 — Presence probability
        prob = _sigmoid(snr, self._seuil_snr, self._sigmoid_scale)
        self._prob_history.append(prob)
        if len(self._prob_history) > self._N_hist:
            self._prob_history = self._prob_history[-self._N_hist:]
        x_prob = np.arange(len(self._prob_history))
        self._line_prob.set_data(x_prob, self._prob_history)
        self._axes[2].set_xlim(0, max(len(self._prob_history), 1))

        # Title feedback
        colour = "green" if detected else "red"
        status = "RESPIRATION DÉTECTÉE" if detected else "Aucune détection"
        self._fig.suptitle(
            f"Radar Micro-Doppler — {status}  |  Trame {n_trame}  |  "
            f"SNR = {snr:.1f} dB  |  P = {prob:.2f}",
            fontsize=13,
            fontweight="bold",
            color=colour,
        )

        return (self._line_rx, self._line_prob)

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
