**English** | [Français](README.fr.md)

# IoT Radar — breathing detection of buried people

A portable **continuous-wave (CW) radar** built on an **ADALM-Pluto** SDR
(AD9363) that detects the breathing of people buried under rubble
(earthquake, collapse, avalanche).  The chest movement modulates the
**phase** of the echo; the software demodulates that phase, turns it into a
chest displacement in millimetres and decides, every half second, whether
someone is breathing, holding their breath (apnea) or moving.

Student project (S6, IoT team).  Everything runs in Python 3.10+, with or
without the radar: a physical simulation and recorded sessions feed exactly
the same processing chain.

> **Status.** The phase chain is validated on the simulated radar (rate within
> ±1 breath/min, displacement amplitude, apnea, empty scene, stream
> discontinuities — see `tests/test_vital_signs_pipeline.py`), including
> people moving around the radar ([Robustness to people moving
> around](#robustness-to-people-moving-around)).  It still has to be validated
> on the PlutoSDR: on the legacy recordings it does not detect breathing, for
> a reason not yet established (see [Known limitations](#known-limitations)).

---

## Contents

1. [Inventory](#inventory)
2. [How it works](#how-it-works)
3. [Robustness to people moving around](#robustness-to-people-moving-around)
4. [Field use: measurement over a silence](#field-use-measurement-over-a-silence)
5. [Repository layout](#repository-layout)
6. [Installation](#installation)
7. [Quick start](#quick-start)
8. [Configuration](#configuration)
9. [Session files (HDF5)](#session-files-hdf5)
10. [Dashboard](#dashboard)
11. [Tests](#tests)
12. [Code organisation rules](#code-organisation-rules)
13. [Known limitations](#known-limitations)

---

## Inventory

State: branch where the component exists (`dev`, `feature/mimo-localization`, `siso-detection`).  Tests: covered by `tests/` of that branch.

### Acquisition and hardware

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| CW transmission | `acquisition/pluto.py` (`cw_tx_buffer`) | cyclic TX buffer, offset snapped to a whole number of periods | | dev | yes |
| Pluto set-up | `acquisition/pluto.py` (`open_pluto`) | pyadi-iio, 1 MS/s, 16 kernel buffers, cyclic transmission | | dev | yes |
| RX checks | `acquisition/pluto.py` | loss register `0x80000088`, saturation warning at 80 % | | dev | yes |
| Pluto source | `acquisition/sources.py` (`PlutoSource`) | time-stamped blocks, losses flagged | | dev | yes |
| Source interface | `acquisition/sources.py` (`Block`, `Source`, `open_source`) | one protocol for Pluto / simulation / replay | | dev | yes |
| 2r2t mode | `acquisition/pluto.py` (`open_pluto_2r2t`) | `adi.ad9361`, 2 coherent RX, 1 or 2 TX, one tone per TX | | feature/mimo-localization, not tested on hardware | yes |
| Localization node | `acquisition/sources.py` (`NodeSetup`, `open_node_source`) | node of `localization.yaml`: 2r2t Pluto or simulated scene | | feature/mimo-localization, not tested on hardware | yes |

### Simulation

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| Simulated CW radar | `acquisition/sources.py` (`CWSimulationSource`) | clutter at f_TX, receiver DC, LO drift, ADC quantisation, timeline | | dev | yes |
| Simulated person | `acquisition/sources.py` (`SimulatedPerson`) | breathing, heartbeat, motion, apnea | | feature/mimo-localization | yes |
| Multi-node scene | `acquisition/sources.py` (`SceneSimulationSource`) | n_tx tones, n_rx coherent channels, reflectors, tones of the other node | | feature/mimo-localization | yes |
| Disturbers | `acquisition/disturbers.py` | walker (torso and limbs), sway, gesture | Chen 2006 | siso-detection | yes |
| Path masking | `acquisition/disturbers.py` (`path_masking`) | down to −20 dB, raised-cosine transition | | siso-detection | yes |
| Static reflectors | `acquisition/sources.py` (`static_scatterers`) | walls masked by the disturbers | | siso-detection | yes |

### Data

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| HDF5 sessions v1 | `acquisition/recording.py` | int16 IQ, blocks, annotations, ground truth, SWMR | | dev | yes |
| Node sessions (v2) | `acquisition/recording.py` | one file per node, geometry, target positions | | feature/mimo-localization | yes |
| Disturbance intervals | `acquisition/recording.py` | `/ground_truth/disturbances`: start, length, kind | | siso-detection | yes |
| Replay | `acquisition/sources.py` (`ReplaySource`) | blocks of a session at the original pace | | dev | yes |
| Legacy `.iq` conversion | `scripts/convert_legacy_iq.py` | `.iq` + `.json` → HDF5 v1 | | dev | yes |
| Calibration files | `localization.py` (`save_calibration`, `load_calibration`) | one YAML per node | | feature/mimo-localization | yes |

### Signal processing

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| Mixer | `dsp/mixer.py` | continuous-phase NCO, echo → 0 Hz | | dev | yes |
| Decimation | `dsp/decimation.py` | Chebyshev I order-8 cascade, prime factors ≤ 13 | | dev | yes |
| Micro-Doppler view | `dsp/spectral.py` | windows, STFT columns (display) | | dev | yes |
| Filters | `dsp/filters.py` | zero-phase Butterworth band-pass, detrend | | dev | yes |
| Circle fit | `dsp/phase.py` (`fit_circle`) | Kåsa then Gauss-Newton | Paterniani 2023 | dev | yes |
| Phase demodulation | `dsp/phase.py` (`demodulate`) | arctangent, DACM or linear, after LO-drift derotation | Paterniani 2023 | dev | yes |
| Doppler filter (M1) | `dsp/doppler.py` | linear-phase Kaiser FIR, −6 dB at 1.5 Hz, delay 1.82 s | Crochiere 1981 | siso-detection | yes |
| Aliasing audit | `dsp/doppler.py` (`alias_audit`) | attenuation of the bands folding into ±1 Hz, per stage and composite | Crochiere 1981 | siso-detection | yes |
| Sliding centre (M2) | `dsp/phase.py` (`track_center`, `demodulate_tracked`) | Pratt on 10 s sub-windows, linear interpolation, off by default | Pratt 1987, Zakrzewski 2012, Park 2007 | siso-detection | yes |
| Contamination (M3) | `dsp/contamination.py` | out-of-band / in-band density per 1 s block, logistic weight, activity | VanderPlas 2018 | siso-detection | yes |
| Multi-channel breathing component | `dsp/doa.py` (`breathing_component`, `tx_phase_reference`) | common band-pass, per-TX phase reference on the leakage | | feature/mimo-localization | yes |

### Estimation and detection

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| Rate estimators | `dsp/estimation.py` | periodogram, FFT peak, ACF, peak count, zero crossings | | dev | yes |
| Breath cycles | `dsp/estimation.py` (`breath_cycles`) | inhalations, intervals, depth, I:E ratio | | dev | yes |
| Detection metrics | `dsp/detection.py` (`breathing_metrics`) | band SNR, peak concentration, motion → confidence | | dev | yes |
| Detector | `dsp/detection.py` (`BreathingDetector`) | smoothing, ON/OFF hysteresis, pause hold, apnea alert | | dev | yes |
| Phase pipeline | `pipeline.py` (`VitalSignsPipeline`) | 20 s window every 0.5 s, restart on discontinuity | | dev | yes |
| Weighted Lomb-Scargle (M3) | `dsp/estimation.py` (`weighted_lomb_scargle`) | floating mean, one offset per segment | VanderPlas 2018 | siso-detection | yes |
| Pause rule (M3) | `pipeline.py` | in-band anomalies cut, blamed on a passer-by or on the target | | siso-detection | yes |
| Measurement mode (M4) | `measurement.py`, `pipeline.py` | 30 s over a silence → decision, rate, quality summary | | siso-detection | yes |
| Multi-channel pipeline | `pipeline.py` (`node_report`) | per-tone demodulation, virtual channels, beamformed channel | | feature/mimo-localization | yes |

### Localization

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| Virtual array | `dsp/doa.py` (`virtual_positions_m`, `steering_vectors`) | elements tx_k + rx_l, far field | | feature/mimo-localization | yes |
| Direction of arrival | `dsp/doa.py` (`estimate_doa`) | Bartlett beamformer, parabolic refinement, Cramér-Rao bound | | feature/mimo-localization, not tested on hardware | yes |
| Channel calibration | `dsp/doa.py` (`estimate_calibration`), `scripts/calibrate_doa.py` | principal eigenvector, target at a known angle | | feature/mimo-localization, not tested on hardware | yes |
| Node geometry | `localization.py` (`NodeGeometry`) | pose, antennas, scene and node frames | | feature/mimo-localization | yes |
| Triangulation | `localization.py` (`triangulate`) | least squares on the angles then Gauss-Newton, χ² ellipse | | feature/mimo-localization, not tested on hardware | yes |
| Node consistency | `localization.py` (`check_consistency`) | rate difference, waveform correlation | | feature/mimo-localization | yes |
| Fusion | `localization.py` (`LocalizationFusion`) | latest report per node, maximum time skew | | feature/mimo-localization | yes |
| Frequency plan | `localization.py` (`check_frequency_plan`) | distance of the tones to harmonics, images and intermodulation | | feature/mimo-localization, not tested on hardware | yes |
| EIRP check | `localization.py` (`estimated_eirp_dbm`) | refuses to transmit above 10 dBm | | feature/mimo-localization, not tested on hardware | yes |

### Physics

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| Link budget | `physics.py` (`radar_range_m`, `range_interval_m`) | monostatic radar equation, pessimistic and optimistic ranges | | dev | yes |

### Machine learning

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| Dataset | `ml/dataset.py` | STFT windows of the legacy `.npz` spectrograms | | dev | yes |
| Autoencoder | `ml/model.py` | Conv2D encoder-decoder + classification head | | dev | yes |
| Training | `ml/train.py`, `scripts/train.py` | AdamW, StepLR, reconstruction + BCE | | dev | yes |

### Interface

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| Phase dashboard | `ui/dashboard.py` | matplotlib, outputs produced in a thread | | dev | yes |
| M3 and measurement indicators | `ui/dashboard.py` | weight, contaminated share, pause cause, key `m` | | siso-detection | yes |
| Home screen | `ui/launcher.py`, `scripts/launcher.py` | pygame, radar started in a subprocess | | dev | yes |
| Localization dashboard | `ui/localization_dashboard.py` | top view, bearings, ellipse, P(θ) spectra | | feature/mimo-localization | yes |

### Tooling

| Component | Location | Method | Reference | State | Tests |
|---|---|---|---|---|---|
| Live radar | `scripts/run_radar.py` | source → pipeline → dashboard or log | | dev | yes |
| Headless measurement | `scripts/run_radar.py` (`--measure`) | one M4 measurement, then exit | | siso-detection | yes |
| Recording | `scripts/record.py` | raw IQ sessions, label, scene | | dev, not tested on hardware | yes |
| Replay | `scripts/replay.py` | session → pipeline → dashboard | | dev | yes |
| Live localization | `scripts/run_localization.py` | one process per node, fusion, display | | feature/mimo-localization, not tested on hardware | yes |
| 2RX hardware check | `scripts/hardware_check_2rx.py` | 2r2t, then 1T2R and 2T2R: tones and phases | | feature/mimo-localization, not tested on hardware | yes |
| Robustness evaluation | `scripts/evaluate_motion_robustness.py` | scenes S1–S8, sessions, Pd/Pfa, κ₀ sweep | | siso-detection | no |
| Configuration | `config.py` | YAML, repository paths, logging | | dev | yes |
| Characterisation references | `tests/characterization_cases.py`, `tests/make_references.py` | DSP outputs compared bit for bit | | dev | yes |
| Dependency rules | `tests/test_architecture.py` | forbidden imports between layers | | dev | yes |
| Fake Pluto | `tests/fake_pluto.py` | pyadi-iio simulated in 2r2t | | feature/mimo-localization | yes |

### Points of attention

- Detection never validated on real data: 2 % of `BREATHING` analyses on the 69 legacy sessions, cause unknown (sample losses of the old 2 MS/s code suspected).
- Full radar (TX + RX + dashboard) at 1 MS/s without loss still to confirm; the dashboard redraws in the acquisition process, 2 MS/s impossible.
- Carrier: `radar.yaml` at 3.5 GHz on `dev` (mobile-operator band in France, not ISM); 2.45 GHz on `siso-detection`, 2.42 GHz for the nodes of `feature/mimo-localization`: to harmonise.
- Localization: hardware chain never checked (Pluto rev. C/D and U.FL, 2r2t mode over SSH, RX1/RX2 and inter-TX phase stability, 600 kS/s × 2 channels without loss).
- EIRP computed from an assumed P(0 dB) ≈ +7 dBm and 6 dBi antennas: to measure (10 mW limit).
- DoA: assumes the TX→RX leakage dominates the clutter; bias of the static reference (~0.3°) not corrected; single target; multipath not handled.
- M3 thresholds (κ₀ 9 dB, activity 6 dB, excursions 12 mm, segments 5 s) tuned on simulation only; disturbers annotated in simulated sessions only.
- Single antenna: a nearby disturber moving within the breathing band (sway at 2 m) cannot be separated from the target.
- M3 without M1 gives false alarms (Pfa 12 % in simulation): keep M1 whenever M3 is enabled.
- Merging both branches: each modifies `sources.py`, `pipeline.py`, `recording.py`, `dashboard.py` and `tests/data/references.npz` (binary) → conflicts, references to regenerate.
- `ml/` obsolete and not connected: it learns from the `.npz` of the old micro-Doppler chain, no longer produced.
- `data/` is ignored by Git: the 69 converted sessions exist on one machine only.
- Plan/code gaps: `MOTION_ROBUSTNESS_PLAN.md` announces a `DopplerLowpass` class (code: function `doppler_lowpass_decimator`).
- FMCW on the Pluto ruled out: USB 2.0 too slow for 20 MS/s, 8.3 m range resolution for 18 MHz of bandwidth, targets masked by the TX→RX leakage.
- `ed_branch` modifies files removed by the refactor (`MicroDopplerDetection/`): conflict if merged.

---

## How it works

### Physics

The Pluto transmits a CW tone at `f_c = 2.45 GHz`, in the ISM band
(wavelength `λ = c / f_c = 12.24 cm`).  A chest moving by `d(t)` towards the radar shifts
the phase of its echo by

```
φ(t) = −4π · d(t) / λ
```

A 10 mm breath gives 1.03 rad; a 0.1 mm heartbeat gives 0.010 rad.  In the
IQ plane, the slow-time samples lie on an **arc of circle** centred on the
static DC offset (TX→RX leakage, walls, receiver DC).  The displacement is
read from the angle **around that centre**, which is why the chain fits the
circle instead of high-pass filtering the IQ (bug B3 of the former chain).

In `cw_offset` mode (default) the tone is transmitted at `f_c + 503.54 Hz`,
so the echo lands away from the receiver DC offset and its 1/f noise.  The
offset is snapped to a whole number of periods per TX buffer (500 Hz
requested → 503.54 Hz transmitted with 65536-sample buffers at 1 MS/s,
bug B1).

### Processing chain

```
PlutoSDR │ simulation │ HDF5 replay        iot_radar/acquisition/sources.py
        │  Block: samples (n_channels, n), sample_start, host_time_s, overflow
        ▼
Mixer (NCO at −tx_offset_hz)               dsp/mixer.py       echo → 0 Hz
        ▼
Decimator (Chebyshev cascade)              dsp/decimation.py  1 MS/s → 100 Hz (Doppler rate)
        ├──► M3 contamination per 1 s block   dsp/contamination.py  weight, activity
        ▼
M1 Doppler low-pass (FIR), ×5              dsp/doppler.py     100 Hz → 20 Hz slow time
        ▼
sliding window: 20 s, analysed every 0.5 s pipeline.py
        ▼
VitalSignsProcessor                        pipeline.py
   LO-drift derotation → circle fit (DC),  dsp/phase.py
     (M2, optional: sliding centre if it drifts)
   → arctangent / DACM → displacement
   → M3 in-band anomalies cut (weight 0)
   → band-pass 0.1–0.5 Hz → spectrum       dsp/filters.py, dsp/estimation.py
     (M3: weighted Lomb-Scargle)
   → breathing metrics (SNR, peak          dsp/detection.py
     concentration, motion)
        ▼
BreathingDetector                          dsp/detection.py
   smoothing, ON/OFF hysteresis, pause hold, median rate, apnea alert
        ▼
M4 measurement mode (30 s, on request)     measurement.py
        ▼
PipelineOutput (dataclasses) ──► dashboard (ui/dashboard.py) or log
```

Each block carries the index of its first sample, the host time and an
overflow flag.  When samples are lost (overflow, or a gap in
`sample_start`), the pipeline resets the oscillator, the filters, the window
and the detector, flags its next output with `discontinuity=True` and starts
again; it never stitches two pieces of signal together.

Detector states: `WARMUP`, `NO_BREATHING`, `BREATHING`, `MOTION` (measurement
paused; `pause_reason` gives the cause: `motion` of the target or ADC
`saturation`).  The apnea alert is raised when no inhalation has been seen
for 10 s although breathing was detected shortly before (a radar cannot tell
a breath-hold from the person leaving the beam).

---

## Robustness to people moving around

In SISO, any movement in the beam used to pause the measurement, so in an
open area the radar was almost always paused.  In the field the search
happens during silences, but residual movement remains (people further
away, someone swaying, gestures).  It is now **removed by processing instead
of pausing** (design and choices: [`MOTION_ROBUSTNESS_PLAN.md`](MOTION_ROBUSTNESS_PLAN.md),
in French).

A radial speed `v` gives the Doppler `f_D = 2·v/λ ≈ 16·v Hz` at 2.45 GHz:

| Source | Doppler at 2.45 GHz |
|---|---|
| breathing | ~0.15 Hz (harmonics up to ~1 Hz) |
| sway of a standing person | ~0.8 Hz |
| gestures | 3–16 Hz |
| walking | 16–25 Hz |

The echoes add **linearly** in the IQ, while the arctangent is not linear:
everything is removed on the complex IQ, **before** the phase demodulation.

- **M1 — Doppler filter** (`dsp/doppler.py`).  Linear-phase FIR low-pass on
  the IQ at 100 Hz: flat to 1 Hz, −6 dB at 1.5 Hz, ≥ 60 dB from 2 Hz, causal,
  constant delay of 1.82 s.  It is also the anti-aliasing filter of the last
  decimation.  The aliasing of the whole chain into ±1 Hz is audited at
  start-up (≥ 84 dB, required 60 dB): a 19 Hz walker never looks like a breath.
- **M2 — sliding centre of the arc** (`dsp/phase.py`).  When people move,
  the static reflections, hence the centre of the IQ arc, drift.  Pratt
  circle fits on 10 s sub-windows every 1 s, interpolated linearly (the
  phase stays continuous).  The tracked centre is only used when it fits
  clearly better than one fixed centre.  **Off by default**: it brought no
  measurable gain in the evaluation, where a passer-by masks a wall only
  briefly (M3 cuts it); enable it where the static clutter drifts slowly.
- **M3 — weights and gap-tolerant estimation** (`dsp/contamination.py`,
  `pipeline.py`).  Per 1 s block, before M1: the contamination index (energy
  density above 2 Hz over the density in the breathing band) gives a smooth
  weight in [0, 1].  The spectrum is a weighted Lomb-Scargle periodogram, so
  contaminated stretches are skipped, and the displayed confidence is
  multiplied by the mean weight.  The measurement pauses **only** for ADC
  saturation or a motion of the target itself.
  - A walker crossing the beam goes through zero radial speed, so for about
    1 s its echo is in the breathing band.  This in-band anomaly is cut out
    (weight 0).  It is blamed on a passer-by if someone was moving out of
    band (3–28 Hz above the receiver noise) in the 10 s before: a person can
    only reach the band by slowing down, so it is seen coming.  Otherwise it
    is the target moving, and the measurement pauses.
- **M4 — measurement mode** (`measurement.py`): see the next section.

M1 and M3 are on by default, M2 is off.  Each of them can be enabled or
disabled in the `motion_robustness` section; with all three disabled, the
former behaviour is restored exactly.

**Evaluation** ([`MOTION_ROBUSTNESS_EVAL.md`](MOTION_ROBUSTNESS_EVAL.md), in
French; simulated scenes and the 69 legacy sessions).  With M1 + M3, against
the former chain:
- the time in pause with people moving around falls from 34–100 % to 0–1 %;
- 30 s measurements: Pd from 58 % to 74 % (walkers crossing at 3 m: from 5 %
  to 70 %), Pfa from 5 % to 0 %;
- a motion of the target still pauses, and its measurements stay
  inconclusive.

---

## Field use: measurement over a silence

1. **Rescuers leave the beam and stand still**: behind the radar or more
   than 60° to the side, nobody within ~3 m in front of it.  A person close
   to the beam whose movements stay in the breathing band (swaying,
   shifting weight) cannot be told apart from the victim by a single
   antenna.
2. **Start the measurement when the silence starts**: key `m` on the
   dashboard, or `python scripts/run_radar.py --measure` without a screen.
   The status card shows `MEASURING x / 30 s`.  The window starts with the
   IQ received after the key press (the delay of M1 is skipped).
3. **Read the decision**:
   - `BREATHING` with the rate;
   - `NO BREATHING`;
   - `INCONCLUSIVE`, with the reason:
     - `target_motion`: the target moved;
     - `saturation`: lower `sdr.rx_gain_db` or move the radar;
     - `too_contaminated`: less than half of the window is usable, someone
       moved in the beam;
     - `discontinuity`: samples were lost;
     - `uncertain`: confidence between the two thresholds.

     Repeat the measurement.

   The quality summary gives the contaminated share, the saturated share
   and the share of time with someone moving around.

The continuous mode keeps running alongside, with the same processing.

---

## Repository layout

```
IoT_radar/
├── iot_radar/                 installable Python package
│   ├── config.py              repository paths, YAML loading, logging
│   ├── physics.py             physical constants, radar range equation
│   ├── acquisition/
│   │   ├── pluto.py           PlutoSDR set-up, TX buffer, overflow / saturation checks
│   │   ├── sources.py         Block, PlutoSource, CWSimulationSource, ReplaySource
│   │   ├── disturbers.py      simulated walkers, sway, gestures, path masking
│   │   └── recording.py       HDF5 session writer and reader (schema v1)
│   ├── dsp/                   signal processing (no I/O, no display)
│   │   ├── mixer.py           NCO frequency shift
│   │   ├── decimation.py      stateful decimator
│   │   ├── doppler.py         M1 Doppler low-pass, aliasing audit
│   │   ├── contamination.py   M3 contamination index, weights, activity
│   │   ├── phase.py           circle fits, M2 sliding centre, arctangent / DACM, phase → displacement
│   │   ├── filters.py         band-pass, detrend
│   │   ├── estimation.py      periodogram, weighted Lomb-Scargle, rate estimators, breath cycles
│   │   ├── detection.py       breathing metrics and BreathingDetector
│   │   └── spectral.py        windows and STFT (micro-Doppler view only)
│   ├── pipeline.py            VitalSignsPipeline: blocks in, PipelineOutput out
│   ├── measurement.py         M4 measurement mode: settings, result, decision rule
│   ├── ui/
│   │   ├── dashboard.py       real-time phase dashboard (matplotlib)
│   │   └── launcher.py        pygame home screen
│   └── ml/                    legacy spectrogram autoencoder (see its README)
├── scripts/                   command-line entry points (see Quick start)
├── configs/                   radar.yaml, training.yaml
├── notebooks/                 inference.ipynb (ml/)
├── tests/                     pytest suite, tests/data/references.npz
├── docs/figures/              figures of the evaluation reports
├── data/sessions/             HDF5 sessions (git-ignored)
├── logs/                      run logs (git-ignored)
├── pyproject.toml, requirements.txt
├── MOTION_ROBUSTNESS_PLAN.md  design of the robustness to residual motion (French)
└── MOTION_ROBUSTNESS_EVAL.md  its evaluation (French)
```

---

## Installation

### System (PlutoSDR only)

```bash
sudo apt update
sudo apt install -y libiio-dev libiio-utils python3-venv
```

```bash
iio_info -s
```

`iio_info -s` must list the Pluto.  Under WSL2, attach the USB device with
`usbipd` first, or use the network address `ip:192.168.2.1`.

### Python

From the repository root:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the tested versions and installs the package in
editable mode (`pip install -e .`).  For a lighter install, choose the
extras you need:

```bash
pip install -e ".[hardware,ui,dev]"
```

| Extra | Packages | Needed for |
|---|---|---|
| *(base)* | numpy, scipy, matplotlib, pyyaml, h5py | processing, dashboard, sessions |
| `hardware` | pyadi-iio, pylibiio | the PlutoSDR |
| `ui` | pygame | the home screen |
| `ml` | torch, tqdm | `scripts/train.py` |
| `notebook` | ipykernel | `notebooks/inference.ipynb` |
| `dev` | pytest | the tests |

---

## Quick start

All commands run from the repository root, with the virtual environment
activated.

**Radar in simulation** (no hardware):

```bash
python scripts/run_radar.py --simulation
```

**Radar on the PlutoSDR:**

```bash
python scripts/run_radar.py
```

Options: `--config FILE`, `--headless` (no window, every result goes to the
log), `--measure` (one measurement over a silence, then exit), `--log-file FILE`.
Logs go to `logs/radar_<timestamp>.log` by default.  On the dashboard, the
key `m` starts a measurement.

**Home screen** with one button per mode (each starts `run_radar.py` in its
own process):

```bash
python scripts/launcher.py
```

**Record a session** (raw IQ, no display).  Two minutes of a breathing person
2.5 m away behind 20 cm of concrete:

```bash
python scripts/record.py --label breathing --duration-s 120 --subject-id S01 --distance-m 2.5 --obstacle concrete --obstacle-thickness-cm 20 --room lab_b12
```

`-n 5 --interval-s 120` records five sessions with a two-minute pause after
each.  With `--simulation`, the label sets the simulated scene, and the
session also stores the simulated chest displacement as ground truth.

**Replay a session** through the pipeline and the dashboard (configuration
stored in the session, `--speed 0` = as fast as possible):

```bash
python scripts/replay.py data/sessions/session_0001_20261005_143012.h5
```

**Convert the legacy `.iq` recordings** into HDF5 sessions
(`data/sessions/legacy/` by default):

```bash
python scripts/convert_legacy_iq.py path/to/old/data
```

**Evaluate the robustness to residual motion** (simulated scenes and
recorded sessions, every combination of M1 to M3; results in
`results/motion_robustness/`, figures in `docs/figures/motion_robustness/`):

```bash
python scripts/evaluate_motion_robustness.py
```

**Train the legacy autoencoder** (see [`iot_radar/ml/README.md`](iot_radar/ml/README.md)):

```bash
python scripts/train.py --epochs 100
```

---

## Configuration

[`configs/radar.yaml`](configs/radar.yaml) is the single settings file of
the radar; every key is commented and its suffix gives its unit (`_hz`,
`_s`, `_m`, `_mm`, `_db`, …).

| Section | Role |
|---|---|
| `logging` | log level, log file on/off |
| `sdr` | Pluto address, carrier, sampling rate, gains, buffer size |
| `tx` | waveform (`cw` / `cw_offset`) and offset |
| `slow_time` | slow-time rate after decimation (20 Hz), warm-up |
| `vital_signs` | analysis window, bands, DC compensation, demodulation, circle-fit validity, `detection` thresholds |
| `motion_robustness` | M1 Doppler filter, aliasing check, M2 centre tracking, M3 contamination weighting (each with `enabled`) |
| `measurement` | M4: duration (30 s), minimum valid share |
| `micro_doppler_view` | waterfall of the dashboard (display only) |
| `display` | dashboard history length, full screen |
| `link_budget` | optimistic / pessimistic scenarios → range shown on the dashboard |
| `recording` | sessions folder, flush interval |
| `simulation` | simulated scene: presence, timeline, breathing, heartbeat, motion, clutter, static reflectors, disturbers, SNR, seed |

A simulated scene can change over time, for example:

```yaml
simulation:
  enabled: true
  timeline: [[0, breathing], [45, apnea], [60, breathing], [95, motion], [110, empty]]
  disturbers:   # people around the radar (still outside their time interval)
    - {kind: walker, start_m: [-8.0, 3.5], velocity_m_s: [1.0, 0.0], start_s: 30.0, stop_s: 46.0}
    - {kind: gesture, position_m: [0.8, 1.5], start_s: 60.0, stop_s: 90.0}
```

**Main parameters of the robustness to motion** (`motion_robustness`):

| Key | Default | Meaning |
|---|---|---|
| `doppler_filter.cutoff_hz` | 1.5 | −6 dB frequency of M1 |
| `doppler_filter.stopband_attenuation_db` | 60 | attenuation from `cutoff + transition/2` = 2 Hz |
| `center_tracking.enabled` | false | M2 (see above) |
| `center_tracking.window_s`, `step_s` | 10, 1 | M2 sub-windows (2 to 3 breathing cycles) |
| `contamination_weighting.weight_midpoint_db` | 9 | contamination index at which the weight is 0.5 |
| `contamination_weighting.activity_threshold_db` | 6 | someone moves around above this |
| `contamination_weighting.activity_lead_s` | 10 | anomaly blamed on a passer-by seen within this delay |
| `contamination_weighting.pause_on_saturation` | true | pause on ADC saturation |

**Sampling rate.**  The Pluto samples at 1 MS/s, with 65536-sample buffers
and 16 RX buffers queued in the Pluto (`sdr.rx_kernel_buffers`, 1 s of
margin).  At 2 MS/s over USB, the PC could not keep up while the dashboard
redrew (same Python process): the Pluto dropped samples and the pipeline
kept restarting.  Losses are logged ("RX samples lost") and shown on the
dashboard as stream discontinuities.  If they come back (slower PC, other
USB link), lower `sample_rate_hz` or raise `rx_kernel_buffers`.

[`configs/training.yaml`](configs/training.yaml) configures `ml/`.

---

## Session files (HDF5)

One file per session, written by `scripts/record.py`
([`acquisition/recording.py`](iot_radar/acquisition/recording.py)) and never
modified after the acquisition.  Name:
`session_<id>_<YYYYMMDD>_<HHMMSS>.h5` (UTC start time); the scene is in the
metadata, not in the name.

```
/                 attributes (below)
/iq               int16 (n_channels, n_samples, 2): I and Q in ADC units
                  (times its "scale" attribute), chunked along time
/blocks           (sample_start, host_time_s, overflow): one row per received block
/annotations      (sample_start, sample_count, label)
/ground_truth/    optional (simulation): time_s, chest_displacement_m
```

| Attributes | |
|---|---|
| radar | `schema_version`, `sample_rate_hz`, `center_frequency_hz`, `tx_waveform`, `tx_offset_hz`, `tx_gain_db`, `rx_gain_db`, `rx_gain_mode`, `n_rx_channels`, `channel_layout`, `firmware_version`, `source_kind`, `iq_stage` |
| session | `start_time_utc`, `software_version`, `config_yaml` (full configuration used) |
| scene | `subject_id`, `distance_m`, `orientation`, `obstacle`, `obstacle_thickness_cm`, `room`, `notes` |

Unknown values are stored as `""` or `NaN`.

Labels: `empty` (nobody in the field), `breathing` (one person, still,
breathing), `apnea` (one person, still, not breathing), `motion` (body
movements dominate), `unknown` (not annotated or uncertain).

The file is written in SWMR mode and flushed every second: if the program is
killed, everything up to the last flush stays readable.  Reading in Python:

```python
from iot_radar.acquisition.recording import SessionReader

with SessionReader("data/sessions/session_0001_20261005_143012.h5") as reader:
    iq = reader.read_iq(0, reader.n_samples)   # complex64 (n_channels, n_samples), ADC units
    print(reader.sample_rate_hz, reader.attributes["distance_m"], reader.annotations())
```

---

## Dashboard

`scripts/run_radar.py` and `scripts/replay.py` open the phase dashboard
([`ui/dashboard.py`](iot_radar/ui/dashboard.py)):

- **status card**: state (and cause of a pause), breathing rate, confidence
  gauge, measurement mode (key `m`: progress, then decision), apnea banner,
  stream discontinuities;
- **displacement waveform** (raw and band-passed, inhalation up) with the
  detected breaths;
- **IQ constellation** of the analysed window with the fitted circle and its
  centre — shows whether the phase can be extracted;
- **confidence timeline** coloured by state, with the M3 weight;
- **measured quantities**: rates of four estimators, breath interval and
  variability, depth, I:E ratio, SNR, echo / DC amplitudes, LO drift,
  indicative heart rate; current weight, contaminated share of the window,
  saturation; summary of the last measurement;
- **rate history**, **displacement spectrum** and **micro-Doppler
  waterfall** of the slow time;
- **parameters** of the radar, the processing and the detection.

The pipeline does not know the dashboard: it returns `PipelineOutput`
dataclasses that the dashboard draws.

---

## Tests

```bash
pytest
```

The suite (about two minutes, no hardware) covers:

- the DSP bricks;
- the sources, including a fake Pluto;
- the HDF5 sessions and the bit-exact live / replay equivalence;
- the validation of the whole chain on the simulated radar;
- the robustness to motion (aliasing ≥ 60 dB, centre tracking, walkers,
  gestures, target motion, saturation, measurement mode);
- the scripts;
- headless smoke tests of the dashboard and the home screen;
- the dependency rules below.

`tests/test_characterization.py` compares DSP outputs bit for bit with
`tests/data/references.npz`.  Regenerate these references
(`python tests/make_references.py`) only when a change of the numerical
results is intended, in a dedicated commit that explains why.

---

## Code organisation rules

Checked by `tests/test_architecture.py`:

| Module | Must not import |
|---|---|
| `dsp/` | `acquisition`, `ui`, `ml`, `pipeline` |
| `pipeline.py` | `ui`, `ml` (`acquisition` only for type hints) |
| `ui/` | `acquisition` (even indirectly), `ml` |
| `ml/` | `acquisition` |
| `config.py`, `physics.py` | anything from the package |

The scripts assemble `acquisition` → `pipeline` → `ui`.  Conventions:
English everywhere, physical names with a unit suffix (`_hz`, `_s`, `_m`,
`_db`, `_rad`), docstrings with shapes and units on the DSP functions.

---

## Known limitations

- **Hardware validation in progress.**  The detection of lost samples
  (register `0x80000088`) works on the Pluto: it revealed the losses at
  2 MS/s.  At 1 MS/s, a receive-only test with dashboard redraws lost no
  samples; this still has to be confirmed with the full radar (TX + RX +
  dashboard), then on new recordings.
- **Legacy recordings.**  The 69 old `.iq` files are converted into
  `data/sessions/legacy/` (`iq_stage = "decimated_clutter_filtered"`).  Their
  high-pass filter only acted around 0 Hz, while the echo sits at +488 Hz,
  so the phase can be re-processed.  Yet the phase chain does **not** detect
  breathing on them:
  - on the `breathing` sessions, only 2 % of the updates are `BREATHING`
    and 57 % are `MOTION` (more than 30 mm peak to peak);
  - on the `empty` sessions, 12 % of the updates are `BREATHING`.

  The cause is not established: moving subjects, a phase too noisy (the
  circle-fit residual is close to that of a noise cloud), or **lost
  samples**.  The old code read the Pluto at 2 MS/s without checking for
  losses, and each loss makes the phase jump.  In the only empty session
  clean enough to check, the phase stays within 0.007 rad between bursts of
  jumps of up to 3 rad, but someone moving in the room would look the same.
  New raw recordings of a controlled scene (`scripts/record.py`) are needed.
- **Robustness to motion: limits of a single antenna** (details in
  `MOTION_ROBUSTNESS_PLAN.md` §3.4 and §7, figures in
  `MOTION_ROBUSTNESS_EVAL.md`):
  - a person near the beam whose movements stay **in the breathing band**
    (swaying, shifting weight), with an echo comparable to the victim's,
    cannot be removed: the measurement is then not conclusive.  Only
    spatial filtering (several antennas, MVDR) could separate them;
  - a disturber that **starts moving directly in the band**, with no
    out-of-band phase first (someone starting to walk next to the target),
    is taken for the target: short pause;
  - if the target moves **while** a passer-by is active, the anomaly is
    blamed on the passer-by: no pause, but the stretch gets weight 0;
  - M1 delays every result by 1.82 s and lengthens the warm-up after a
    restart (3.6 s);
  - M3 without M1 lets the disturbers reach the demodulation and gives
    false detections in an empty room: keep M1 enabled with M3;
  - the thresholds (index midpoint, activity threshold) are set on the
    simulation; they must be checked on new recordings with annotated
    disturbers (`/ground_truth/disturbances` is written for simulated
    sessions only).
- **`ml/`** still learns from the legacy `.npz` spectrograms of the removed
  micro-Doppler chain; it will be redesigned on the phase signal of the HDF5
  sessions.
- **FMCW** (range-gated breathing) was studied but is not part of this
  package; see [Points of attention](#points-of-attention).

---

## Licence

Internal project — academic and research use.
