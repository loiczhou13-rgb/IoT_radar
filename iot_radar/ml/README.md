**English** | [Français](README.fr.md)

# `iot_radar.ml` — legacy spectrogram autoencoder

> **Status: legacy, to be redesigned.**  This sub-package still learns from
> the `.npz` **micro-Doppler spectrograms** recorded by the former chain.
> That chain has been removed, so **nothing in the repository produces these
> files any more**, and the old `.npz` dataset is no longer kept in the
> repository.  The code has only been moved and translated during the
> refactor, not redesigned.  It will be redesigned to take the **phase
> signal** (chest displacement) computed from the HDF5 sessions as input.

## What it contains

| File | Role |
|---|---|
| `dataset.py` | `CalibrationDataset`: loads every `.npz` under a folder and cuts each spectrogram into windows of `n_cols` STFT columns, shape `(1, n_fft, n_cols)`, with the label (0 = empty, 1 = breathing) |
| `model.py` | `SpectrogramAutoencoder`: Conv2D encoder / decoder and a binary classification head on the latent |
| `train.py` | training loop (library part): composite loss `α · reconstruction + (1 − α) · BCE`, AdamW, StepLR, best checkpoint, CSV history |

Entry points: [`scripts/train.py`](../../scripts/train.py) and
[`notebooks/inference.ipynb`](../../notebooks/inference.ipynb) (training
curves, confusion matrix, reconstructions).  Every hyper-parameter is in
[`configs/training.yaml`](../../configs/training.yaml).

## Running it

Requires the `ml` extra (`pip install -e ".[ml]"`) and a copy of the legacy
`.npz` files under `data/ml/` (git-ignored):

```
data/ml/
├── train/<n>.npz
└── val/<n>.npz      optional; otherwise val_split of train/
```

Each `.npz` holds at least `spectrogram_db` (`(n_frames, n_fft)`, dB),
`label`, `env`, `n_fft` and `f_s_dec_hz`.  Default input shape:
`(1, 8192, 32)`, latent `(128, 32, 2)`.

```bash
python scripts/train.py --epochs 100
```

Options: `--config`, `--epochs`, `--batch-size`, `--lr`, `--device`
(`auto` / `cpu` / `cuda`).  Outputs go to `results/` (git-ignored):
`best.pt` (weights and embedded configuration), `history.csv`, `train.log`.

## Planned redesign

- **Input:** windows of the chest displacement (or of the compensated slow-time
  IQ) computed by `iot_radar.pipeline` from the HDF5 sessions, instead of
  spectrograms of the high-passed IQ.
- **Labels:** the `/annotations` of the sessions (`empty`, `breathing`,
  `apnea`, `motion`, `unknown`).
- **Data:** the training script reads the sessions (`ReplaySource`) and
  runs them through the pipeline, so that training sees exactly what the
  real-time chain sees.

The dependency rule stays: `ml/` must not import `acquisition/` (checked by
`tests/test_architecture.py`); `scripts/train.py` does the assembly.
