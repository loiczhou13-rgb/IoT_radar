"""Real-time dashboard of the phase-demodulation radar (matplotlib).

Place in the chain: last step.  The dashboard displays the
:class:`iot_radar.pipeline.PipelineOutput` objects it receives; it does not
know where the samples come from (live radar, simulation or replay).

Layout (GridSpec 3 × 5)::

    ┌──────────┬──────────────────────────────┬────────────┬────────────┐
    │ status   │ displacement waveform +       │ IQ         │            │
    │ card     │ breath marks                  │ constell.  │            │
    ├──────────┴──────────────────────────────┼────────────┤ parameters │
    │ confidence timeline (decision, instant., │ measured   │ (radar,    │
    │ SNR / concentration scores, states)      │ quantities │ processing,│
    ├──────────┬───────────────┬───────────────┤            │ detection) │
    │ rate     │ displacement  │ micro-Doppler │            │            │
    │ history  │ spectrum      │ waterfall     │            │            │
    └──────────┴───────────────┴───────────────┴────────────┴────────────┘

* **Status card** — detector state, breathing rate, confidence gauge with the
  ON/OFF thresholds, duration of the breathing episode, apnea banner, stream
  discontinuities.
* **Displacement waveform** — chest expansion towards the radar (inhalation
  up), raw and band-passed, with end-of-inhalation ▲ / end-of-exhalation ▼.
* **IQ constellation** — the analysed window in the IQ plane with the fitted
  circle and its centre (DC offset): shows whether the phase can be
  extracted (arc length, noise, clutter).
* **Confidence timeline** — smoothed and instantaneous confidence, SNR and
  concentration scores; background coloured by state.
* **Measured quantities** — rates of four estimators, breath interval and
  variability, depth, I:E ratio, time since the last breath, SNR,
  concentration, echo / DC amplitudes, LO drift, indicative heart rate.
* **Rate history**, **displacement spectrum** (bands, peak window, noise
  floor) and **micro-Doppler waterfall** of the slow time.
* **Parameters** — static radar, processing and detection parameters.

Threading model: the outputs are produced in a daemon thread (blocking SDR
reads never freeze the window); a matplotlib timer moves them to the
display every 100 ms, so none is lost.  Two WSLg + TkAgg pitfalls are
avoided: the producer starts only after the first ``draw_event`` (window
painted), and the output iterator is only closed by the producer thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from iot_radar.dsp.detection import (
    STATE_BREATHING,
    STATE_CODES,
    STATE_MOTION,
    STATE_NO_BREATHING,
    STATE_WARMUP,
)

if TYPE_CHECKING:  # type hints only: the dashboard does not depend on the pipeline
    from iot_radar.dsp.detection import BreathingState
    from iot_radar.pipeline import PipelineOutput, WindowAnalysis

logger = logging.getLogger(__name__)

_POLL_MS = 100
_APNEA_COLOR = "#de2d26"
_DISCONTINUITY_COLOR = "#756bb1"

STATE_STYLE = {
    STATE_WARMUP: ("#9ecae1", "WARMING UP"),
    STATE_NO_BREATHING: ("#bdbdbd", "NO BREATHING"),
    STATE_BREATHING: ("#31a354", "BREATHING DETECTED"),
    STATE_MOTION: ("#fd8d3c", "MOTION — MEASUREMENT PAUSED"),
}
"""Colour and label of each detector state."""

_TEXT_BOX = dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#bbbbbb")


def _table(sections: list[tuple[str, list[tuple[str, str]]]], key_width: int = 19) -> str:
    """Format ``[(title, [(key, value), ...]), ...]`` as aligned monospace text."""
    lines: list[str] = []
    for title, rows in sections:
        if lines:
            lines.append("")
        lines.append(f"── {title} " + "─" * max(0, 30 - len(title)))
        lines += [f" {key:<{key_width}}{value}" for key, value in rows]
    return "\n".join(lines)


def _fmt(value: float | None, fmt: str, unit: str = "", default: str = "—") -> str:
    """Format a possibly missing value."""
    if value is None or not np.isfinite(value):
        return default
    return f"{value:{fmt}}{unit}"


class PhaseDashboard:
    """Matplotlib dashboard of the breathing detection.

    Parameters
    ----------
    context : dict
        Static description of the run:

        * ``title`` and ``subtitle`` (str);
        * ``micro_doppler_f_hz`` (axis of the micro-Doppler columns, Hz) and
          ``display_max_hz`` (half-span shown);
        * ``waterfall_columns`` (int) and ``history_s`` (length of the
          timelines, s);
        * ``breathing_band_hz``, ``heart_band_hz``, ``reference_band_hz``,
          ``threshold_on``, ``threshold_off``, ``peak_halfwidth_hz``;
        * ``parameters`` — ``[(section title, [(name, value), ...]), ...]``.
    full_screen : bool, optional
        Toggle the full-screen mode at start-up.
    """

    def __init__(self, context: dict[str, Any], full_screen: bool = False) -> None:
        self._context = context
        f_hz = np.asarray(context["micro_doppler_f_hz"], dtype=np.float64)
        self._waterfall_rows = np.abs(f_hz) <= float(context.get("display_max_hz", 3.0))
        self._waterfall_f_hz = f_hz[self._waterfall_rows]
        self._n_columns = int(context.get("waterfall_columns", 240))
        self._breathing_band_hz = tuple(context.get("breathing_band_hz", (0.1, 0.5)))
        self._reference_band_hz = tuple(context.get("reference_band_hz", (2.5, 5.0)))
        self._heart_band_hz = tuple(context.get("heart_band_hz", (0.8, 2.0)))
        self._threshold_on = float(context.get("threshold_on", 0.6))
        self._threshold_off = float(context.get("threshold_off", 0.4))
        self._peak_halfwidth_hz = float(context.get("peak_halfwidth_hz", 0.1))
        self._history_s = float(context.get("history_s", 120.0))

        self._waterfall = np.full((self._waterfall_f_hz.size, self._n_columns), np.nan)
        self._history: deque[tuple[float, ...]] = deque()
        self._discontinuity_times_s: deque[float] = deque()
        self._pending: deque[PipelineOutput] = deque(maxlen=200)
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None
        self._color_limits_set = False
        self._state_fills: list = []
        self._discontinuity_lines: list = []

        self.fig = plt.figure(figsize=(20, 10.5), constrained_layout=True)
        title = context.get("title", "Radar breathing detection")
        if context.get("subtitle"):
            title += "\n" + context["subtitle"]
        self.fig.suptitle(title, fontsize=12, fontweight="bold")
        grid = GridSpec(3, 5, figure=self.fig, width_ratios=(1.0, 1.1, 1.1, 1.0, 1.0))
        self.ax_status = self.fig.add_subplot(grid[0, 0])
        self.ax_wave = self.fig.add_subplot(grid[0, 1:3])
        self.ax_iq = self.fig.add_subplot(grid[0, 3])
        self.ax_confidence = self.fig.add_subplot(grid[1, 0:3])
        self.ax_live = self.fig.add_subplot(grid[1:, 3])
        self.ax_rate = self.fig.add_subplot(grid[2, 0])
        self.ax_spectrum = self.fig.add_subplot(grid[2, 1])
        self.ax_waterfall = self.fig.add_subplot(grid[2, 2])
        self.ax_parameters = self.fig.add_subplot(grid[:, 4])
        self._init_panels()

        if full_screen:
            try:
                plt.get_current_fig_manager().full_screen_toggle()
            except Exception:
                pass
        logger.info("Dashboard ready (backend %s)", matplotlib.get_backend())

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    def _init_panels(self) -> None:
        self._init_status()

        # --- Displacement waveform ------------------------------------------
        ax = self.ax_wave
        (self._line_raw,) = ax.plot([], [], color="0.75", linewidth=0.8, label="raw")
        (self._line_breath,) = ax.plot([], [], color="tab:blue", linewidth=1.8, label="breathing band")
        (self._marks_inhalation,) = ax.plot([], [], linestyle="none", marker="^", color="tab:green",
                                            markersize=8, label="end of inhalation")
        (self._marks_exhalation,) = ax.plot([], [], linestyle="none", marker="v", color="tab:purple",
                                            markersize=7, label="end of exhalation")
        ax.set_xlabel("Time in the analysis window (s)")
        ax.set_ylabel("Chest expansion (mm)")
        ax.set_title("Breathing waveform (inhalation ↑, towards the radar)")
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.grid(alpha=0.3)

        # --- IQ constellation -----------------------------------------------
        ax = self.ax_iq
        self._iq_scatter = ax.scatter([], [], s=6, c=[], cmap="viridis", vmin=0, vmax=1)
        (self._iq_last,) = ax.plot([], [], "o", color="tab:red", markersize=6)
        self._iq_circle = Circle((0, 0), 1, fill=False, linestyle="--", color="tab:red", linewidth=1.2)
        self._iq_circle.set_visible(False)
        ax.add_patch(self._iq_circle)
        (self._iq_center,) = ax.plot([], [], "+", color="tab:red", markersize=12, mew=2)
        self._iq_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=8, va="bottom",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title("IQ constellation (colour = time)")
        ax.grid(alpha=0.3)

        # --- Confidence timeline --------------------------------------------
        ax = self.ax_confidence
        (self._line_snr_score,) = ax.plot([], [], color="tab:orange", linewidth=0.9, linestyle="--",
                                          label="SNR score")
        (self._line_concentration_score,) = ax.plot([], [], color="tab:cyan", linewidth=0.9,
                                                    linestyle="--", label="concentration score")
        (self._line_instant,) = ax.plot([], [], color="0.55", linewidth=0.9, label="instantaneous")
        (self._line_confidence,) = ax.plot([], [], color="black", linewidth=2.2, label="smoothed (decision)")
        ax.axhline(self._threshold_on, color="tab:green", linestyle="--", linewidth=1.0,
                   label=f"ON ({self._threshold_on:.2f})")
        ax.axhline(self._threshold_off, color="tab:red", linestyle=":", linewidth=1.0,
                   label=f"OFF ({self._threshold_off:.2f})")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-self._history_s, 0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Confidence / score")
        ax.set_title("Detection confidence — background: green breathing · orange motion · "
                     "red apnea alert · purple line: stream discontinuity")
        ax.legend(loc="lower left", fontsize=8, ncol=6)
        ax.grid(alpha=0.3)

        # --- Text panels ------------------------------------------------------
        for ax, title in ((self.ax_live, "Measured quantities"), (self.ax_parameters, "Parameters")):
            ax.set_axis_off()
            ax.set_title(title, fontsize=10, fontweight="bold")
        self._live_text = self.ax_live.text(0.0, 1.0, "Waiting for the first analysis window…",
                                            transform=self.ax_live.transAxes, va="top",
                                            fontsize=8, fontfamily="monospace", bbox=_TEXT_BOX)
        self.ax_parameters.text(0.0, 1.0, _table(self._context.get("parameters", [])),
                                transform=self.ax_parameters.transAxes, va="top", fontsize=8,
                                fontfamily="monospace", bbox=_TEXT_BOX)

        # --- Rate history ---------------------------------------------------
        ax = self.ax_rate
        (self._line_rate_fft,) = ax.plot([], [], linestyle="none", marker=".", color="0.6",
                                         markersize=4, label="per window (FFT)")
        (self._line_rate_cycles,) = ax.plot([], [], linestyle="none", marker="x", color="tab:blue",
                                            markersize=3, label="per window (cycles)")
        (self._line_rate,) = ax.plot([], [], color="tab:green", linewidth=2.2, label="reported")
        ax.axhspan(60 * self._breathing_band_hz[0], 60 * self._breathing_band_hz[1], color="tab:green", alpha=0.07)
        ax.set_ylim(0, 60 * self._breathing_band_hz[1] * 1.3)
        ax.set_xlim(-self._history_s, 0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Breaths / min")
        ax.set_title("Breathing rate")
        ax.legend(loc="lower left", fontsize=7)
        ax.grid(alpha=0.3)

        # --- Displacement spectrum ---------------------------------------------
        ax = self.ax_spectrum
        ax.axvspan(*self._breathing_band_hz, color="tab:green", alpha=0.10, label="breathing band")
        ax.axvspan(*self._heart_band_hz, color="tab:pink", alpha=0.06, label="heart band")
        ax.axvspan(*self._reference_band_hz, color="tab:red", alpha=0.06, label="noise reference")
        (self._line_spectrum,) = ax.plot([], [], color="tab:purple", linewidth=1.0)
        self._peak_span = ax.axvspan(0, 0, color="tab:green", alpha=0.35, label="peak window")
        (self._line_noise_floor,) = ax.plot([], [], color="tab:red", linestyle="--", linewidth=1.0,
                                            label="noise floor")
        ax.set_xlim(0, self._reference_band_hz[1] + 0.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_title("Displacement spectrum")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)

        # --- Micro-Doppler waterfall -----------------------------------------
        ax = self.ax_waterfall
        f_hz = self._waterfall_f_hz
        self._waterfall_image = ax.imshow(
            self._waterfall, aspect="auto", origin="lower", cmap="viridis",
            extent=(-self._n_columns, 0, f_hz[0], f_hz[-1]), interpolation="nearest",
        )
        ax.set_xlabel("Updates")
        ax.set_ylabel("Doppler (Hz)")
        ax.set_title("Micro-Doppler spectrogram of the slow time (dB)", fontsize=10)

    def _init_status(self) -> None:
        """Status card: state, rate, confidence gauge, episode duration, apnea banner."""
        ax = self.ax_status
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        self._card = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.0,rounding_size=0.04",
            transform=ax.transAxes, facecolor=STATE_STYLE[STATE_WARMUP][0], alpha=0.25,
            edgecolor="none",
        )
        ax.add_patch(self._card)
        self._state_text = ax.text(0.5, 0.88, STATE_STYLE[STATE_WARMUP][1], ha="center",
                                   va="center", fontsize=13, fontweight="bold")
        self._rate_text = ax.text(0.5, 0.69, "— /min", ha="center", va="center",
                                  fontsize=28, fontweight="bold")
        ax.text(0.08, 0.50, "Confidence", fontsize=9, va="center")
        self._confidence_text = ax.text(0.92, 0.50, "0 %", fontsize=12, fontweight="bold",
                                        ha="right", va="center")
        ax.add_patch(Rectangle((0.08, 0.36), 0.84, 0.09, facecolor="#eeeeee", edgecolor="#999999"))
        self._confidence_bar = Rectangle((0.08, 0.36), 0.0, 0.09, facecolor=STATE_STYLE[STATE_WARMUP][0])
        ax.add_patch(self._confidence_bar)
        for threshold, label in ((self._threshold_off, "OFF"), (self._threshold_on, "ON")):
            x = 0.08 + 0.84 * threshold
            ax.plot([x, x], [0.34, 0.47], color="black", linewidth=1.0)
            ax.text(x, 0.31, label, ha="center", va="top", fontsize=7)
        self._episode_text = ax.text(0.5, 0.20, "", ha="center", va="center", fontsize=10)
        self._alert_text = ax.text(0.5, 0.08, "", ha="center", va="center", fontsize=10,
                                   fontweight="bold", color="white",
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor=_APNEA_COLOR))
        self._alert_text.set_visible(False)

    # ------------------------------------------------------------------
    # Update with one output (GUI thread)
    # ------------------------------------------------------------------

    def update(self, output: PipelineOutput) -> None:
        """Refresh every panel with one pipeline output."""
        if output.discontinuity:
            self._discontinuity_times_s.append(output.t_s)
        self._update_waterfall(output.micro_doppler_column_db)
        self._update_status(output.breathing, output.t_s)
        self._update_timelines(output.breathing, output.analysis)
        if output.analysis is not None:
            self._update_waveform(output.analysis)
            self._update_constellation(output.analysis)
            self._update_spectrum(output.analysis)
            self._live_text.set_text(self._measured_quantities(output.breathing, output.analysis))

    def _update_waterfall(self, column_db: np.ndarray | None) -> None:
        if column_db is None:
            return
        column_db = np.asarray(column_db)[self._waterfall_rows]
        self._waterfall = np.roll(self._waterfall, -1, axis=1)
        self._waterfall[:, -1] = column_db
        self._waterfall_image.set_data(self._waterfall)
        if not self._color_limits_set and np.isfinite(column_db).any():
            top_db = float(np.nanmax(column_db))
            self._waterfall_image.set_clim(top_db - 60.0, top_db)
            self._color_limits_set = True

    def _update_status(self, breathing: BreathingState, t_s: float) -> None:
        color, label = STATE_STYLE.get(breathing.state, ("#bdbdbd", breathing.state))
        self._card.set_facecolor(_APNEA_COLOR if breathing.apnea else color)
        self._state_text.set_text(label)
        self._state_text.set_color(color if breathing.state != STATE_NO_BREATHING else "0.35")
        self._rate_text.set_text(f"{breathing.rate_bpm:.1f} /min" if breathing.rate_bpm else "— /min")
        self._confidence_text.set_text(f"{100 * breathing.confidence:.0f} %")
        self._confidence_bar.set_width(0.84 * float(np.clip(breathing.confidence, 0, 1)))
        self._confidence_bar.set_facecolor(color)
        if breathing.state == STATE_BREATHING:
            minutes, seconds = divmod(int(breathing.breathing_for_s), 60)
            self._episode_text.set_text(f"Breathing for {minutes:02d}:{seconds:02d}")
        elif breathing.state == STATE_MOTION:
            self._episode_text.set_text("Rate paused until the window is clean")
        else:
            self._episode_text.set_text("")

        recent_restart = self._discontinuity_times_s and t_s - self._discontinuity_times_s[-1] < 5.0
        if breathing.apnea:
            self._alert_text.set_text(
                f"NO BREATH FOR {breathing.since_last_breath_s:.0f} s (apnea or target lost)")
            self._alert_text.get_bbox_patch().set_facecolor(_APNEA_COLOR)
        elif recent_restart:
            self._alert_text.set_text("SAMPLES LOST — analysis restarted")
            self._alert_text.get_bbox_patch().set_facecolor(_DISCONTINUITY_COLOR)
        self._alert_text.set_visible(bool(breathing.apnea or recent_restart))

    def _update_timelines(self, breathing: BreathingState, analysis: WindowAnalysis | None) -> None:
        metrics = breathing.metrics
        reliable = metrics is not None and metrics.confidence >= self._threshold_off
        rate_fft_bpm = 60 * metrics.rate_hz if reliable and metrics.rate_hz else np.nan
        rate_cycles_bpm = (60 * analysis.cycles.rate_hz
                           if reliable and analysis is not None and analysis.cycles.rate_hz else np.nan)
        self._history.append((
            breathing.t_s,
            breathing.confidence,
            breathing.confidence_instant,
            STATE_CODES[breathing.state],
            breathing.rate_bpm if breathing.rate_bpm is not None else np.nan,
            metrics.snr_score if metrics is not None else np.nan,
            metrics.concentration_score if metrics is not None else np.nan,
            float(breathing.apnea),
            rate_fft_bpm,
            rate_cycles_bpm,
        ))
        while self._history and self._history[0][0] < breathing.t_s - self._history_s:
            self._history.popleft()
        history = np.array(self._history, dtype=np.float64)
        t_s = history[:, 0] - breathing.t_s
        self._line_confidence.set_data(t_s, history[:, 1])
        self._line_instant.set_data(t_s, history[:, 2])
        self._line_snr_score.set_data(t_s, history[:, 5])
        self._line_concentration_score.set_data(t_s, history[:, 6])
        self._line_rate.set_data(t_s, history[:, 4])
        self._line_rate_fft.set_data(t_s, history[:, 8])
        self._line_rate_cycles.set_data(t_s, history[:, 9])

        for artist in self._state_fills + self._discontinuity_lines:
            artist.remove()
        self._state_fills, self._discontinuity_lines = [], []
        backgrounds = (
            (history[:, 3] == STATE_CODES[STATE_BREATHING], STATE_STYLE[STATE_BREATHING][0]),
            (history[:, 3] == STATE_CODES[STATE_MOTION], STATE_STYLE[STATE_MOTION][0]),
            (history[:, 7] > 0, _APNEA_COLOR),
        )
        for mask, color in backgrounds:
            if mask.any():
                self._state_fills.append(self.ax_confidence.fill_between(
                    t_s, 0, 1, where=mask, step="post", color=color, alpha=0.18, linewidth=0,
                ))
        for restart_s in self._discontinuity_times_s:
            if restart_s >= breathing.t_s - self._history_s:
                self._discontinuity_lines.append(self.ax_confidence.axvline(
                    restart_s - breathing.t_s, color=_DISCONTINUITY_COLOR, linewidth=1.5))

    def _update_waveform(self, analysis: WindowAnalysis) -> None:
        unit = "mm" if analysis.demodulation.calibrated else "a.u."
        chest_raw = -analysis.displacement_mm
        chest = -analysis.breath_mm
        self._line_raw.set_data(analysis.t_s, chest_raw)
        self._line_breath.set_data(analysis.t_s, chest)
        cycles = analysis.cycles
        self._marks_inhalation.set_data(analysis.t_s[cycles.peak_indices], chest[cycles.peak_indices])
        self._marks_exhalation.set_data(analysis.t_s[cycles.trough_indices], chest[cycles.trough_indices])
        last_interval = f"{cycles.intervals_s[-1]:.1f} s" if cycles.intervals_s.size else "—"
        self.ax_wave.set_title(
            f"Breathing waveform (inhalation ↑) — {cycles.peak_indices.size} breaths in window · "
            f"last interval {last_interval} · depth {_fmt(cycles.depth, '.1f', ' ' + unit)}")
        self.ax_wave.set_ylabel(f"Chest expansion ({unit})")
        self.ax_wave.relim()
        self.ax_wave.autoscale_view()

    def _update_constellation(self, analysis: WindowAnalysis) -> None:
        z = analysis.iq
        self._iq_scatter.set_offsets(np.column_stack((z.real, z.imag)))
        self._iq_scatter.set_array(np.linspace(0, 1, z.size))
        self._iq_last.set_data([z[-1].real], [z[-1].imag])
        circle = analysis.demodulation.circle
        xs = [z.real.min(), z.real.max()]
        ys = [z.imag.min(), z.imag.max()]
        if circle is not None and circle.radius > 0 and circle.valid:
            self._iq_circle.set_center((circle.center.real, circle.center.imag))
            self._iq_circle.set_radius(circle.radius)
            self._iq_circle.set_visible(True)
            self._iq_center.set_data([circle.center.real], [circle.center.imag])
            xs += [circle.center.real - circle.radius, circle.center.real + circle.radius]
            ys += [circle.center.imag - circle.radius, circle.center.imag + circle.radius]
            offset_to_echo_db = 20 * np.log10(max(abs(circle.center), 1e-12) / circle.radius)
            self._iq_text.set_text(
                f"arc {np.degrees(circle.arc_span_rad):.0f}° · residual {100 * circle.rms_residual:.0f} %\n"
                f"|C|/A = {offset_to_echo_db:.1f} dB")
        else:
            self._iq_circle.set_visible(False)
            self._iq_center.set_data([], [])
            self._iq_text.set_text("circle fit rejected → linear demodulation")
        half_span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-12) * 0.6
        center_x = 0.5 * (min(xs) + max(xs))
        center_y = 0.5 * (min(ys) + max(ys))
        self.ax_iq.set_xlim(center_x - half_span, center_x + half_span)
        self.ax_iq.set_ylim(center_y - half_span, center_y + half_span)

    def _update_spectrum(self, analysis: WindowAnalysis) -> None:
        metrics = analysis.metrics
        f_hz = analysis.spectrum_f_hz
        shown = f_hz <= self._reference_band_hz[1] + 0.5
        psd_db = 10.0 * np.log10(analysis.spectrum_psd[shown] + 1e-300)
        self._line_spectrum.set_data(f_hz[shown], psd_db)
        floor_db = 10.0 * np.log10(metrics.noise_floor)
        self._line_noise_floor.set_data([0, f_hz[shown][-1]], [floor_db, floor_db])
        top_db = float(psd_db.max())
        self.ax_spectrum.set_ylim(min(floor_db - 15.0, top_db - 40.0), top_db + 5.0)
        if metrics.rate_hz is not None:
            low_hz = max(self._breathing_band_hz[0], metrics.rate_hz - self._peak_halfwidth_hz)
            high_hz = min(self._breathing_band_hz[1], metrics.rate_hz + self._peak_halfwidth_hz)
            self._peak_span.set_x(low_hz)  # axvspan returns a Rectangle (matplotlib ≥ 3.9)
            self._peak_span.set_width(high_hz - low_hz)
        self.ax_spectrum.set_title(f"Displacement spectrum — SNR {metrics.snr_db:.1f} dB · "
                                   f"conc. {metrics.concentration:.2f}")

    def _measured_quantities(self, breathing: BreathingState, analysis: WindowAnalysis) -> str:
        """Text of the measured-quantities panel."""
        metrics, cycles, demodulation = analysis.metrics, analysis.cycles, analysis.demodulation
        unit = "mm" if demodulation.calibrated else "a.u."
        rates = analysis.rates_hz

        def bpm(rate_hz: float | None) -> str:
            return _fmt(60 * rate_hz if rate_hz else None, ".1f")

        if cycles.intervals_s.size and np.isfinite(cycles.variability):
            interval = f"{cycles.intervals_s.mean():.2f} s (CV {100 * cycles.variability:.0f} %)"
        else:
            interval = _fmt(cycles.intervals_s.mean() if cycles.intervals_s.size else None, ".2f", " s")
        breathing_rows = [
            ("Reported rate", _fmt(breathing.rate_bpm, ".1f", " /min")),
            ("FFT | ACF", f"{bpm(rates.get('fft'))} | {bpm(rates.get('acf'))} /min"),
            ("Cycles | ZC", f"{bpm(rates.get('peaks'))} | {bpm(rates.get('zero_crossing'))} /min"),
            ("Breath interval", interval),
            ("Depth", _fmt(cycles.depth, ".1f", f" {unit}")),
            ("I:E ratio", _fmt(cycles.ie_ratio, ".2f") if demodulation.calibrated else "— (sign unknown)"),
            ("Last breath", _fmt(breathing.since_last_breath_s, ".1f", " s ago")),
        ]
        detection_rows = [
            ("Band SNR", f"{metrics.snr_db:.1f} dB ({100 * metrics.snr_score:.0f} %)"),
            ("Concentration", f"{metrics.concentration:.2f} ({100 * metrics.concentration_score:.0f} %)"),
            ("Confidence i | s", f"{100 * metrics.confidence:.0f} % | {100 * breathing.confidence:.0f} %"),
            ("Peak genuine", "yes" if metrics.genuine_peak else "no (band-edge skirt)"),
            ("Motion p-p", _fmt(metrics.displacement_ptp_mm, ".1f", " mm", "n/a (linear)")),
        ]
        signal_rows = [("Demodulation", demodulation.method)]
        if demodulation.circle is not None and demodulation.circle.valid:
            signal_rows += [
                ("Echo amplitude A", f"{demodulation.circle.radius:.3g}"),
                ("DC offset |C|", f"{abs(demodulation.circle.center):.3g}"),
            ]
        signal_rows += [
            ("LO drift", f"{analysis.lo_drift_hz * 1e3:+.1f} mHz"),
            ("Heart rate (ind.)", bpm(analysis.heart_rate_hz) + " /min"),
        ]
        return _table([("Breathing", breathing_rows), ("Detection", detection_rows), ("Signal", signal_rows)])

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def run(self, outputs: Iterable[PipelineOutput]) -> None:
        """Consume *outputs* in a producer thread and display them until the window is closed."""

        def _produce() -> None:
            try:
                for output in outputs:
                    if self._stop.is_set():
                        break
                    self._pending.append(output)
                    time.sleep(0)  # yield the GIL to the Tk main loop
            except Exception:
                logger.exception("The output producer failed")
            finally:
                close = getattr(outputs, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
                logger.info("Output producer stopped")

        def _start(_event=None) -> None:
            if self._producer is None:
                self._producer = threading.Thread(target=_produce, name="radar-producer", daemon=True)
                self._producer.start()

        def _poll() -> None:
            updated = False
            while self._pending:
                self.update(self._pending.popleft())
                updated = True
            if updated:
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect("draw_event", _start)
        self.fig.canvas.mpl_connect("close_event", lambda _event: self._stop.set())
        timer = self.fig.canvas.new_timer(interval=_POLL_MS)
        timer.add_callback(_poll)
        timer.start()
        try:
            plt.show()
        finally:
            self._stop.set()
            if self._producer is not None and self._producer.is_alive():
                self._producer.join(timeout=2.0)
