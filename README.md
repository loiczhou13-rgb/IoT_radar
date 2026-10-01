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

1. [How it works](#how-it-works)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [Session files (HDF5)](#session-files-hdf5)
7. [Dashboard](#dashboard)
8. [Tests](#tests)
9. [Code organisation rules](#code-organisation-rules)
10. [Known limitations](#known-limitations)

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

In `cw_offset` mode (default) the tone is transmitted at `f_c + 488.28 Hz`,
so the echo lands away from the receiver DC offset and its 1/f noise.  The
offset is snapped to a whole number of periods per TX buffer (500 Hz
requested → 488.28 Hz transmitted, bug B1).

### Processing chain

```
PlutoSDR │ simulation │ HDF5 replay        iot_radar/acquisition/sources.py
        │  Block: samples (n_channels, n), sample_start, host_time_s, overflow
        ▼
Mixer (NCO at −tx_offset_hz)               dsp/mixer.py       echo → 0 Hz
        ▼
Decimator (Chebyshev cascade)              dsp/decimation.py  2 MS/s → 20 Hz slow time
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

- **No validation on new hardware recordings yet.**  The PlutoSDR overflow
  detection (register `0x80000088`) and the real-time chain on the Pluto
  have not been tested since the refactor.
- **Legacy recordings.**  The 69 old `.iq` files are converted into
  `data/sessions/legacy/` (`iq_stage = "decimated_clutter_filtered"`).  Their
  high-pass filter only acted around 0 Hz, while the echo sits at +488 Hz,
  so the phase can be re-processed.  Yet the phase chain does **not** detect
  breathing on them:
  - on the `breathing` sessions, only 2 % of the updates are `BREATHING`
    and 57 % are `MOTION` (more than 30 mm peak to peak);
  - on the `empty` sessions, 12 % of the updates are `BREATHING`.

  The cause is not established: moving subjects, or a phase too noisy (the
  circle-fit residual is close to that of a noise cloud).  New raw
  recordings of a controlled scene (`scripts/record.py`) are needed.  Details
  in section 19.3 of `REFACTOR_PLAN.md`.
- **`ml/`** still learns from the legacy `.npz` spectrograms of the removed
  micro-Doppler chain; it will be redesigned on the phase signal of the HDF5
  sessions.
- **FMCW** (range-gated breathing) was studied but is not part of this
  package; see the "FMCW: findings" section of `REFACTOR_PLAN.md`.

---

## Licence

Internal project — academic and research use.
