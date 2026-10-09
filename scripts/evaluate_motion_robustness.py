#!/usr/bin/env python3
"""Evaluate the robustness to residual motion: former chain against every combination of M1 to M3.

Protocol of MOTION_ROBUSTNESS_PLAN.md §6.  Each configuration processes:

* simulated scenes S1–S8 (target, walkers, sway, gestures, empty room,
  target moving), several seeds, at a reduced input rate (20 kS/s, same
  decimation from 100 Hz on) and at 2.45 GHz;
* the annotated HDF5 sessions of a folder (the 69 converted legacy sessions
  by default, labelled ``breathing`` or ``empty``).

Measured, per configuration and scene: the share of the time in pause
(``MOTION`` outputs of the continuous chain), the decisions of the
measurement mode (M4) on back-to-back 30 s windows — Pd = P(breathing |
target), Pfa = P(breathing | no target), share inconclusive — and the rate
error of the windows decided ``breathing`` (simulation).

Results in ``results/motion_robustness/results.json`` and
``summary.md``; figures in ``docs/figures/motion_robustness/``.

Examples (from the repository root)::

    python scripts/evaluate_motion_robustness.py                 # everything (≈ 10 min on 24 cores)
    python scripts/evaluate_motion_robustness.py --quick         # 1 seed, 90 s, 10 sessions
    python scripts/evaluate_motion_robustness.py --no-sessions --jobs 8
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from iot_radar.acquisition.pluto import ADC_FULL_SCALE, effective_tx_offset_hz
from iot_radar.acquisition.sources import ReplaySource, open_source
from iot_radar.config import DEFAULT_RADAR_CONFIG, load_config
from iot_radar.pipeline import VitalSignsPipeline

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "motion_robustness"
FIGURES_DIR = ROOT / "docs" / "figures" / "motion_robustness"

METHODS = ("doppler_filter", "center_tracking", "contamination_weighting")
CONFIGS: dict[str, tuple[str, ...]] = {
    "former": (),
    "M1": ("doppler_filter",),
    "M2": ("center_tracking",),
    "M3": ("contamination_weighting",),
    "M1+M2": ("doppler_filter", "center_tracking"),
    "M1+M3": ("doppler_filter", "contamination_weighting"),
    "M2+M3": ("center_tracking", "contamination_weighting"),
    "M1+M2+M3": METHODS,
}
"""Methods enabled in each compared configuration ("former": the pause on any movement)."""

SCENES: dict[str, str] = {
    "S1": "target alone",
    "S2": "target + walkers crossing at 2.5–3.5 m",
    "S3": "target + walkers crossing at 8 m",
    "S4": "target + sway of a witness at 2 m",
    "S5": "target + gestures at 1.5 m",
    "S6": "empty room + walkers at 2.5–3.5 m",
    "S7": "empty room + gestures at 1.5 m",
    "S8": "target moving itself (3 × 6 s)",
}
TARGET_SCENES = ("S1", "S2", "S3", "S4", "S5")
EMPTY_SCENES = ("S6", "S7")
TARGET_MOTION_S = ((20.0, 26.0), (70.0, 76.0), (120.0, 126.0))

SIMULATION_RATE_HZ = 20e3
TX_OFFSET_HZ = 244.140625      # 25 periods per 2048-sample buffer at 20 kS/s
SNR_PER_SAMPLE_DB = -5.0       # ≈ 25 dB in the 20 Hz slow time
CROSSING_PERIOD_S = 25.0       # one walker crossing the beam every 25 s
WALLS = [{"position_m": [1.5, 6.0], "amplitude": 3.0}, {"position_m": [-2.0, 5.0], "amplitude": 2.0}]
"""Static reflectors of every simulated scene: the passers-by mask them, so the centre of the arc moves."""


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def _crossings(depths_m: tuple[float, ...], duration_s: float, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Walkers crossing the beam every CROSSING_PERIOD_S, alternating direction and depth.

    Each one walks from x = ∓9 m to x = ±9 m at 0.5–1.5 m/s, then stays
    where it stopped (a person does not vanish).
    """
    walkers = []
    for k, start_s in enumerate(np.arange(2.0, duration_s, CROSSING_PERIOD_S)):
        speed = float(rng.uniform(0.5, 1.5))
        side = -1.0 if k % 2 == 0 else 1.0
        walkers.append({"kind": "walker", "start_m": [9.0 * side, depths_m[k % len(depths_m)]],
                        "velocity_m_s": [-side * speed, 0.0], "start_s": float(start_s),
                        "stop_s": float(start_s + 18.0 / speed)})
    return walkers


def scene_simulation(scene: str, seed: int, duration_s: float) -> dict[str, Any]:
    """``simulation`` section of a scene: breathing drawn from the seed, disturbers of the scene."""
    rng = np.random.default_rng(1000 + seed)
    simulation: dict[str, Any] = {
        "enabled": True, "realtime": False, "seed": seed, "snr_db": SNR_PER_SAMPLE_DB,
        "breath_rate_hz": float(rng.uniform(0.2, 0.4)), "breath_amplitude_mm": float(rng.uniform(2.5, 5.0)),
        "target_range_m": 3.0, "presence": scene not in EMPTY_SCENES, "disturbers": [],
        "static_scatterers": WALLS,
    }
    walker_rng = np.random.default_rng(2000 + seed)
    if scene in ("S2", "S6"):
        simulation["disturbers"] = _crossings((3.5, 2.5), duration_s, walker_rng)
    elif scene == "S3":
        simulation["disturbers"] = _crossings((8.0,), duration_s, walker_rng)
    elif scene == "S4":
        simulation["disturbers"] = [{"kind": "sway", "position_m": [1.0, 1.7]}]
    elif scene in ("S5", "S7"):
        simulation["disturbers"] = [{"kind": "gesture", "position_m": [0.8, 1.3]}]
    elif scene == "S8":
        timeline: list[list[Any]] = [[0.0, "breathing"]]
        for start_s, stop_s in TARGET_MOTION_S:
            timeline += [[start_s, "motion"], [stop_s, "breathing"]]
        simulation["timeline"] = timeline
    return simulation


def configured(base: dict[str, Any], methods: tuple[str, ...]) -> dict[str, Any]:
    """*base* with exactly the methods *methods* of ``motion_robustness`` enabled."""
    cfg = copy.deepcopy(base)
    section = cfg.setdefault("motion_robustness", {})
    for method in METHODS:
        section.setdefault(method, {})["enabled"] = method in methods
    return cfg


def simulation_base() -> dict[str, Any]:
    """Default configuration at the reduced rate of the simulated scenes (2.45 GHz)."""
    cfg = load_config(DEFAULT_RADAR_CONFIG)
    cfg["sdr"].update(sample_rate_hz=SIMULATION_RATE_HZ, buffer_size=2048, center_frequency_hz=2.45e9)
    cfg["tx"]["offset_hz"] = TX_OFFSET_HZ
    return cfg


# ---------------------------------------------------------------------------
# One stream through one configuration
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    """What one stream gives with one configuration."""

    pause_fraction: float
    measurements: list[dict[str, Any]]
    trace: dict[str, list] = field(default_factory=dict)


def run_stream(cfg: dict[str, Any], source, tx_offset_hz: float, adc_full_scale: float | None,
               duration_s: float | None = None, keep_trace: bool = False) -> StreamResult:
    """Continuous chain and back-to-back measurements (M4) over the whole stream."""
    pipeline = VitalSignsPipeline.from_config(cfg, source.sample_rate_hz, tx_offset_hz, adc_full_scale)
    n_analysed = n_paused = 0
    measurements: list[dict[str, Any]] = []
    trace: dict[str, list] = {"t_s": [], "state": [], "weight": [], "confidence": [], "column_db": []}
    pipeline.start_measurement()
    try:
        while True:
            block = source.read_block()
            if block is None:
                break
            outputs = pipeline.process_block(block)
            for output in outputs:
                if output.measurement.state == "done":
                    result = output.measurement.result
                    measurements.append({
                        "t_start_s": result.t_start_s, "t_end_s": result.t_end_s, "decision": result.decision,
                        "reasons": list(result.reasons), "rate_bpm": result.rate_bpm,
                        "confidence": result.confidence, "contaminated": result.contaminated_fraction})
                if output.analysis is not None:
                    n_analysed += 1
                    n_paused += output.breathing.state == "MOTION"
                if keep_trace:
                    trace["t_s"].append(output.t_s)
                    trace["state"].append(output.breathing.state)
                    trace["weight"].append(output.contamination.weight if output.contamination else 1.0)
                    trace["confidence"].append(output.breathing.confidence)
                    column = output.micro_doppler_column_db
                    trace["column_db"].append(None if column is None else column.tolist())
            if not pipeline.measuring:
                pipeline.start_measurement()          # windows back to back
            if duration_s is not None and outputs and outputs[-1].t_s >= duration_s:
                break
    finally:
        source.close()
    if keep_trace:
        trace["f_hz"] = pipeline.micro_doppler_f_hz.tolist()
    return StreamResult(n_paused / max(n_analysed, 1), measurements, trace)


def simulated_job(scene: str, seed: int, config: str, duration_s: float, keep_trace: bool = False,
                  midpoint_db: float | None = None) -> dict[str, Any]:
    """One simulated scene, seed and configuration (run in a worker process).

    *midpoint_db* overrides the index at which the M3 weight is 0.5 (sweep of κ₀).
    """
    logging.disable(logging.WARNING)
    cfg = configured(simulation_base(), CONFIGS[config])
    if midpoint_db is not None:
        cfg["motion_robustness"]["contamination_weighting"]["weight_midpoint_db"] = midpoint_db
    cfg["simulation"].update(scene_simulation(scene, seed, duration_s))
    source = open_source(cfg, simulation=True)
    result = run_stream(cfg, source, effective_tx_offset_hz(cfg), ADC_FULL_SCALE, duration_s, keep_trace)
    out = {"kind": "simulation" if midpoint_db is None else "midpoint_sweep", "midpoint_db": midpoint_db,
           "scene": scene, "seed": seed, "config": config,
           "true_rate_bpm": 60.0 * cfg["simulation"]["breath_rate_hz"], "target": scene not in EMPTY_SCENES,
           "pause_fraction": result.pause_fraction, "measurements": result.measurements}
    if keep_trace:
        out["trace"] = result.trace
        source = open_source(cfg, simulation=True)
        out["disturbances_s"] = [(start / source.sample_rate_hz, (start + count) / source.sample_rate_hz)
                                 for start, count, _ in source.disturbance_intervals(
                                     int(duration_s * source.sample_rate_hz))]
        source.close()
    return out


def session_job(path: str, config: str) -> dict[str, Any]:
    """One recorded session and configuration (run in a worker process)."""
    logging.disable(logging.WARNING)
    source = ReplaySource(Path(path), speed=None)
    labels = sorted({label for _, _, label in source.annotations()})
    cfg = configured(load_config(DEFAULT_RADAR_CONFIG), CONFIGS[config])
    cfg["sdr"]["center_frequency_hz"] = float(source.attributes["center_frequency_hz"])
    raw_adc = source.attributes.get("iq_stage") == "raw_adc"
    result = run_stream(cfg, source, float(source.attributes["tx_offset_hz"]),
                        ADC_FULL_SCALE if raw_adc else None)
    return {"kind": "session", "session": Path(path).name, "config": config, "labels": labels,
            "target": "breathing" in labels, "pause_fraction": result.pause_fraction,
            "measurements": result.measurements}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _overlaps_target_motion(measurement: dict[str, Any]) -> bool:
    return any(measurement["t_start_s"] < stop + 2.0 and measurement["t_end_s"] > start
               for start, stop in TARGET_MOTION_S)


def decision_rates(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Shares of breathing / no_breathing / inconclusive decisions over the windows of *runs*."""
    decisions = [m["decision"] for run in runs for m in run["measurements"]]
    n = max(len(decisions), 1)
    return {"n": len(decisions), **{d: decisions.count(d) / n for d in ("breathing", "no_breathing", "inconclusive")}}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Tables of the report: pause share, Pd / Pfa, inconclusive share, rate error."""
    summary: dict[str, Any] = {"pause": {}, "simulation": {}, "sessions": {}, "rate_error": {}, "s8": {}}
    simulated = [r for r in results if r["kind"] == "simulation"]
    sessions = [r for r in results if r["kind"] == "session"]
    for config in CONFIGS:
        mine = [r for r in simulated if r["config"] == config]
        summary["pause"][config] = {
            scene: float(np.mean([r["pause_fraction"] for r in mine if r["scene"] == scene]))
            for scene in SCENES if any(r["scene"] == scene for r in mine)}
        target = decision_rates([r for r in mine if r["scene"] in TARGET_SCENES])
        empty = decision_rates([r for r in mine if r["scene"] in EMPTY_SCENES])
        summary["simulation"][config] = {
            "Pd": target["breathing"], "inconclusive_target": target["inconclusive"], "n_target": target["n"],
            "Pfa": empty["breathing"], "inconclusive_empty": empty["inconclusive"], "n_empty": empty["n"],
            "Pd_by_scene": {scene: decision_rates([r for r in mine if r["scene"] == scene])["breathing"]
                            for scene in TARGET_SCENES}}
        errors = [abs(m["rate_bpm"] - r["true_rate_bpm"]) for r in mine if r["target"]
                  for m in r["measurements"] if m["decision"] == "breathing"]
        summary["rate_error"][config] = {"n": len(errors), "median": float(np.median(errors)) if errors else None,
                                         "p95": float(np.percentile(errors, 95)) if errors else None,
                                         "errors": errors}
        s8 = [m for r in mine if r["scene"] == "S8" for m in r["measurements"]]
        moving = [m for m in s8 if _overlaps_target_motion(m)]
        summary["s8"][config] = {
            "pause": summary["pause"][config].get("S8"),
            "inconclusive_when_moving": (float(np.mean([m["decision"] == "inconclusive" for m in moving]))
                                         if moving else None),
            "n_moving": len(moving)}
        own = [r for r in sessions if r["config"] == config]
        if own:
            target = decision_rates([r for r in own if r["target"]])
            empty = decision_rates([r for r in own if not r["target"]])
            summary["sessions"][config] = {
                "pause_target": float(np.mean([r["pause_fraction"] for r in own if r["target"]])),
                "pause_empty": float(np.mean([r["pause_fraction"] for r in own if not r["target"]])),
                "Pd": target["breathing"], "inconclusive_target": target["inconclusive"], "n_target": target["n"],
                "Pfa": empty["breathing"], "inconclusive_empty": empty["inconclusive"], "n_empty": empty["n"]}
    return summary


def summarize_sweep(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Pd, Pfa, inconclusive share and pause share of M1+M2+M3 for each κ₀ of the sweep."""
    sweep = [r for r in results if r["kind"] == "midpoint_sweep"]
    table = {}
    for midpoint_db in sorted({r["midpoint_db"] for r in sweep}):
        mine = [r for r in sweep if r["midpoint_db"] == midpoint_db]
        target = decision_rates([r for r in mine if r["target"]])
        empty = decision_rates([r for r in mine if not r["target"]])
        table[f"{midpoint_db:g}"] = {"Pd": target["breathing"], "inconclusive_target": target["inconclusive"],
                                     "Pfa": empty["breathing"], "inconclusive_empty": empty["inconclusive"],
                                     "pause": float(np.mean([r["pause_fraction"] for r in mine]))}
    return table


def markdown_tables(summary: dict[str, Any]) -> str:
    """Markdown tables of the summary (pasted into MOTION_ROBUSTNESS_EVAL.md)."""
    scenes = list(next(iter(summary["pause"].values())).keys())
    lines = ["### Part du temps en pause (%)", "",
             "| Config. | " + " | ".join(scenes) + " |", "|---|" + "---|" * len(scenes)]
    for config, row in summary["pause"].items():
        lines.append(f"| {config} | " + " | ".join(f"{100 * row[s]:.0f}" for s in scenes) + " |")
    lines += ["", "### Mode mesure sur fenêtres de 30 s — simulation", "",
              "| Config. | Pd | non conclu (cible) | Pfa | non conclu (vide) | erreur médiane / P95 (/min) |",
              "|---|---|---|---|---|---|"]
    for config, row in summary["simulation"].items():
        error = summary["rate_error"][config]
        error_text = f"{error['median']:.2f} / {error['p95']:.2f}" if error["median"] is not None else "—"
        lines.append(f"| {config} | {100 * row['Pd']:.0f} % | {100 * row['inconclusive_target']:.0f} % | "
                     f"{100 * row['Pfa']:.0f} % | {100 * row['inconclusive_empty']:.0f} % | {error_text} |")
    lines += ["", "### Pd par scène (%)", "", "| Config. | " + " | ".join(TARGET_SCENES) + " |",
              "|---|" + "---|" * len(TARGET_SCENES)]
    for config, row in summary["simulation"].items():
        lines.append(f"| {config} | " + " | ".join(f"{100 * row['Pd_by_scene'][s]:.0f}" for s in TARGET_SCENES)
                     + " |")
    lines += ["", "### S8 — la cible bouge", "", "| Config. | pause | fenêtres avec mouvement non conclues |",
              "|---|---|---|"]
    for config, row in summary["s8"].items():
        moving = row["inconclusive_when_moving"]
        lines.append(f"| {config} | {100 * (row['pause'] or 0):.0f} % | "
                     f"{'—' if moving is None else f'{100 * moving:.0f} % ({row['n_moving']})'} |")
    if summary["sessions"]:
        lines += ["", "### Sessions enregistrées", "",
                  "| Config. | pause (respiration / vide) | Pd | non conclu | Pfa | non conclu (vide) |",
                  "|---|---|---|---|---|---|"]
        for config, row in summary["sessions"].items():
            lines.append(f"| {config} | {100 * row['pause_target']:.0f} % / {100 * row['pause_empty']:.0f} % | "
                         f"{100 * row['Pd']:.0f} % | {100 * row['inconclusive_target']:.0f} % | "
                         f"{100 * row['Pfa']:.0f} % | {100 * row['inconclusive_empty']:.0f} % |")
    if summary.get("midpoint_sweep"):
        lines += ["", "### Sensibilité à κ₀ (M1+M2+M3, S2, S5, S6, S7)", "",
                  "| κ₀ (dB) | Pd | non conclu (cible) | Pfa | non conclu (vide) | pause |", "|---|---|---|---|---|---|"]
        for midpoint, row in summary["midpoint_sweep"].items():
            lines.append(f"| {midpoint} | {100 * row['Pd']:.0f} % | {100 * row['inconclusive_target']:.0f} % | "
                         f"{100 * row['Pfa']:.0f} % | {100 * row['inconclusive_empty']:.0f} % | "
                         f"{100 * row['pause']:.0f} % |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_figures(summary: dict[str, Any], examples: dict[str, dict[str, Any]], figures_dir: Path) -> None:
    """Pause shares, Pd / Pfa, rate errors and a time example of S2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    configs = list(CONFIGS)

    # Pause share: configurations × scenes.
    scenes = list(summary["pause"][configs[0]].keys())
    values = np.array([[100 * summary["pause"][c][s] for s in scenes] for c in configs])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    image = ax.imshow(values, cmap="Oranges", vmin=0, vmax=max(50.0, values.max()))
    for i in range(len(configs)):
        for j in range(len(scenes)):
            ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(scenes)), scenes)
    ax.set_yticks(range(len(configs)), configs)
    ax.set_title("Part du temps en pause (%) — simulation")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(figures_dir / "pause_share.png", dpi=120)
    plt.close(fig)

    # Pd / Pfa / inconclusive per configuration (simulation and sessions).
    panels = [("Simulation", summary["simulation"])]
    if summary["sessions"]:
        panels.append(("Sessions enregistrées", summary["sessions"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), constrained_layout=True, squeeze=False)
    x = np.arange(len(configs))
    for ax, (title, table) in zip(axes[0], panels):
        for k, (key, label, color) in enumerate((("Pd", "Pd", "tab:green"),
                                                 ("inconclusive_target", "non conclu (cible)", "tab:olive"),
                                                 ("Pfa", "Pfa", "tab:red"))):
            ax.bar(x + (k - 1) * 0.27, [100 * table[c][key] for c in configs], 0.27, label=label, color=color)
        ax.set_xticks(x, configs, rotation=30, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% des fenêtres de 30 s")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(figures_dir / "pd_pfa.png", dpi=120)
    plt.close(fig)

    # Rate error of the windows decided "breathing".
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    data = [summary["rate_error"][c]["errors"] or [np.nan] for c in configs]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(configs) + 1), configs, rotation=30, fontsize=8)
    ax.set_ylabel("|erreur| (respirations/min)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Erreur de fréquence des fenêtres décidées « respiration » — simulation")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(figures_dir / "rate_error.png", dpi=120)
    plt.close(fig)

    # Time example of S2, former chain against M1+M2+M3.
    if examples:
        state_colors = {"WARMUP": "#9ecae1", "NO_BREATHING": "#bdbdbd", "BREATHING": "#31a354", "MOTION": "#fd8d3c"}
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True,
                                 gridspec_kw={"height_ratios": (2, 2, 1.3)})
        top = None
        for ax, config in zip(axes[:2], ("former", "M1+M2+M3")):
            trace = examples[config]["trace"]
            t_s = np.array(trace["t_s"])
            f_hz = np.array(trace["f_hz"])
            shown = np.abs(f_hz) <= 10.0
            columns = np.array([c if c is not None else [np.nan] * f_hz.size for c in trace["column_db"]]).T[shown]
            top = np.nanmax(columns) if top is None else top       # same colour scale for both
            ax.imshow(columns, aspect="auto", origin="lower", cmap="viridis", vmin=top - 60, vmax=top,
                      extent=(t_s[0], t_s[-1], f_hz[shown][0], f_hz[shown][-1]))
            for state, color in state_colors.items():
                mask = np.array(trace["state"]) == state
                ax.fill_between(t_s, -10, -8.8, where=mask, step="post", color=color, linewidth=0)
            ax.set_ylabel("Doppler du temps lent (Hz)")
            ax.set_title(f"{config} — micro-Doppler à 20 Hz et état (bande du bas : vert respiration, "
                         "orange pause, gris non détecté)", fontsize=9)
        trace = examples["M1+M2+M3"]["trace"]
        axes[2].plot(trace["t_s"], trace["weight"], color="#8c6d31", label="poids M3 du dernier bloc")
        axes[2].plot(trace["t_s"], trace["confidence"], color="black", label="confiance lissée (M1+M2+M3)")
        axes[2].plot(examples["former"]["trace"]["t_s"], examples["former"]["trace"]["confidence"],
                     color="0.6", label="confiance lissée (référence)")
        for start_s, stop_s in examples["M1+M2+M3"].get("disturbances_s", []):
            axes[2].axvspan(start_s + 1.82, stop_s + 1.82, color="tab:red", alpha=0.1)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_xlabel("Temps (s) — en rouge pâle : marcheur dans le champ (vérité terrain, + retard de M1)")
        axes[2].legend(fontsize=8, loc="lower right")
        axes[2].grid(alpha=0.3)
        fig.savefig(figures_dir / "example_s2.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description="Evaluate the robustness to residual motion (M1 to M4).")
    p.add_argument("--seeds", type=int, default=5, help="seeds per simulated scene (default: %(default)s)")
    p.add_argument("--duration-s", type=float, default=150.0, help="length of a simulated scene (s)")
    p.add_argument("--sessions", type=Path, default=ROOT / "data" / "sessions" / "legacy",
                   help="folder of annotated HDF5 sessions (default: %(default)s)")
    p.add_argument("--no-sessions", action="store_true", help="simulation only")
    p.add_argument("--max-sessions", type=int, default=None, help="only the first N sessions")
    p.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    p.add_argument("--quick", action="store_true", help="1 seed, 90 s, 10 sessions (smoke run)")
    p.add_argument("--midpoints-db", type=float, nargs="*", default=[],
                   help="also sweep κ₀ (weight_midpoint_db) of M1+M2+M3 on S2, S5, S6 and S7, e.g. 0 3 6 9 12")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run every job, write the results, the summary tables and the figures."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.quick:
        args.seeds, args.duration_s, args.max_sessions = 1, 90.0, args.max_sessions or 10
    jobs: list[tuple] = [("sim", scene, seed, config, args.duration_s)
                         for scene in SCENES for seed in range(args.seeds) for config in CONFIGS]
    sessions = [] if args.no_sessions else sorted(args.sessions.glob("*.h5"))[:args.max_sessions]
    jobs += [("session", str(path), config) for path in sessions for config in CONFIGS]
    jobs += [("sweep", scene, seed, "M1+M2+M3", args.duration_s, False, midpoint_db)
             for midpoint_db in args.midpoints_db for scene in ("S2", "S5", "S6", "S7") for seed in range(args.seeds)]
    logger.info("%d jobs (%d simulated, %d on %d sessions)", len(jobs), len(SCENES) * args.seeds * len(CONFIGS),
                len(sessions) * len(CONFIGS), len(sessions))

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(session_job if job[0] == "session" else simulated_job, *job[1:]) for job in jobs]
        examples = {config: pool.submit(simulated_job, "S2", 0, config, args.duration_s, True)
                    for config in ("former", "M1+M2+M3")}
        for k, future in enumerate(futures, 1):
            results.append(future.result())
            if k % 50 == 0 or k == len(futures):
                logger.info("%d / %d jobs done", k, len(futures))
        examples = {config: future.result() for config, future in examples.items()}

    summary = summarize(results)
    summary["midpoint_sweep"] = summarize_sweep(results)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "results.json").write_text(json.dumps({"results": results, "summary": summary}, indent=1),
                                                   encoding="utf-8")
    tables = markdown_tables(summary)
    (args.results_dir / "summary.md").write_text(tables, encoding="utf-8")
    plot_figures(summary, examples, args.figures_dir)
    print(tables)
    return summary


if __name__ == "__main__":
    main()
