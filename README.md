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
> discontinuities — see `tests/test_vital_signs_pipeline.py`).  It still has
> to be validated on the PlutoSDR: on the legacy recordings it does not
> detect breathing, for a reason not yet established (see
> [Known limitations](#known-limitations)).

---

## Contents

1. [Inventory](#inventory)
2. [How it works](#how-it-works)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Configuration](#configuration)
7. [Session files (HDF5)](#session-files-hdf5)
8. [Dashboard](#dashboard)
9. [Tests](#tests)
10. [Code organisation rules](#code-organisation-rules)
11. [Known limitations](#known-limitations)

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
- Plan/code gaps: `MOTION_ROBUSTNESS_PLAN.md` announces a `DopplerLowpass` class (code: function `doppler_lowpass_decimator`); B9 of `REFACTOR_PLAN.md` (`ip:` throughput at 2 MS/s) is void since the move to 1 MS/s.
- FMCW on the Pluto ruled out: USB 2.0 too slow for 20 MS/s, 8.3 m range resolution for 18 MHz of bandwidth, targets masked by the TX→RX leakage.
- `ed_branch` modifies files removed by the refactor (`MicroDopplerDetection/`): conflict if merged.

---

## How it works

### Physics

The Pluto transmits a CW tone at `f_c = 3.5 GHz` (wavelength
`λ = c / f_c = 8.57 cm`).  A chest moving by `d(t)` towards the radar shifts
the phase of its echo by

```
φ(t) = −4π · d(t) / λ
```

A 10 mm breath gives 1.47 rad; a 0.1 mm heartbeat gives 0.015 rad.  In the
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
Decimator (Chebyshev cascade)              dsp/decimation.py  1 MS/s → 20 Hz slow time
        ▼
sliding window: 20 s, analysed every 0.5 s pipeline.py
        ▼
VitalSignsProcessor                        pipeline.py
   LO-drift derotation → circle fit (DC)   dsp/phase.py
   → arctangent / DACM → displacement
   → band-pass 0.1–0.5 Hz → spectrum       dsp/filters.py, dsp/estimation.py
   → breathing metrics (SNR, peak          dsp/detection.py
     concentration, motion)
        ▼
BreathingDetector                          dsp/detection.py
   smoothing, ON/OFF hysteresis, MOTION hold, median rate, apnea alert
        ▼
PipelineOutput (dataclasses) ──► dashboard (ui/dashboard.py) or log
```

Each block carries the index of its first sample, the host time and an
overflow flag.  When samples are lost (overflow, or a gap in
`sample_start`), the pipeline resets the oscillator, the filters, the window
and the detector, flags its next output with `discontinuity=True` and starts
again; it never stitches two pieces of signal together.

Detector states: `WARMUP`, `NO_BREATHING`, `BREATHING`, `MOTION`.  The apnea alert
is raised when no inhalation has been seen for 10 s although breathing was
detected shortly before (a radar cannot tell a breath-hold from the person
leaving the beam).

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
│   │   └── recording.py       HDF5 session writer and reader (schema v1)
│   ├── dsp/                   signal processing (no I/O, no display)
│   │   ├── mixer.py           NCO frequency shift
│   │   ├── decimation.py      stateful decimator
│   │   ├── phase.py           circle fit, arctangent / DACM, phase → displacement
│   │   ├── filters.py         band-pass, detrend
│   │   ├── estimation.py      periodogram, rate estimators, breath-by-breath cycles
│   │   ├── detection.py       breathing metrics and BreathingDetector
│   │   └── spectral.py        windows and STFT (micro-Doppler view only)
│   ├── pipeline.py            VitalSignsPipeline: blocks in, PipelineOutput out
│   ├── ui/
│   │   ├── dashboard.py       real-time phase dashboard (matplotlib)
│   │   └── launcher.py        pygame home screen
│   └── ml/                    legacy spectrogram autoencoder (see its README)
├── scripts/                   command-line entry points (see Quick start)
├── configs/                   radar.yaml, training.yaml
├── notebooks/                 inference.ipynb (ml/)
├── tests/                     pytest suite, tests/data/references.npz
├── data/sessions/             HDF5 sessions (git-ignored)
├── logs/                      run logs (git-ignored)
├── pyproject.toml, requirements.txt
└── REFACTOR_PLAN.md           design decisions of the 2026 refactor (French)
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
log), `--log-file FILE`.  Logs go to `logs/radar_<timestamp>.log` by default.

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
| `micro_doppler_view` | waterfall of the dashboard (display only) |
| `display` | dashboard history length, full screen |
| `link_budget` | optimistic / pessimistic scenarios → range shown on the dashboard |
| `recording` | sessions folder, flush interval |
| `simulation` | simulated scene: presence, timeline, breathing, heartbeat, motion, clutter, SNR, seed |

A simulated scene can change over time, for example:

```yaml
simulation:
  enabled: true
  timeline: [[0, breathing], [45, apnea], [60, breathing], [95, motion], [110, empty]]
```

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

- **status card**: state, breathing rate, confidence gauge, apnea banner,
  stream discontinuities;
- **displacement waveform** (raw and band-passed, inhalation up) with the
  detected breaths;
- **IQ constellation** of the analysed window with the fitted circle and its
  centre — shows whether the phase can be extracted;
- **confidence timeline** coloured by state;
- **measured quantities**: rates of four estimators, breath interval and
  variability, depth, I:E ratio, SNR, echo / DC amplitudes, LO drift,
  indicative heart rate;
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

The suite (under a minute, no hardware) covers:

- the DSP bricks;
- the sources, including a fake Pluto;
- the HDF5 sessions and the bit-exact live / replay equivalence;
- the validation of the whole chain on the simulated radar;
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
  Details in section 19.3 of `REFACTOR_PLAN.md`.
- **`ml/`** still learns from the legacy `.npz` spectrograms of the removed
  micro-Doppler chain; it will be redesigned on the phase signal of the HDF5
  sessions.
- **FMCW** (range-gated breathing) was studied but is not part of this
  package; see the "FMCW: findings" section of `REFACTOR_PLAN.md`.

---

## Licence

Internal project — academic and research use.
