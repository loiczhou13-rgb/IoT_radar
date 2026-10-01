"""Behaviour tests of the calibration dataset, the autoencoder and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from iot_radar.ml.dataset import CalibrationDataset
from iot_radar.ml.model import SpectrogramAutoencoder
from script_loader import load_script

N_FFT = 16
SMALL_MODEL = {
    "expected_input_shape": [1, N_FFT, 32],
    "encoder_channels": [1, 4, 8, 8],
    "downsample_factors": [[2, 2], [2, 2]],
    "in_conv_kernel": [3, 3],
    "refine_kernel": [3, 3],
    "out_conv_kernel": [3, 3],
    "classifier": {"hidden": 8, "dropout": 0.0},
}


def _write_recording(path: Path, n_frames: int, label: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    spectrogram_db = rng.normal(-40.0 + 10.0 * label, 3.0, size=(n_frames, N_FFT))
    np.savez_compressed(path, spectrogram_db=spectrogram_db, label=np.int8(label), env=np.asarray("salle"))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "train"
    root.mkdir(parents=True)
    _write_recording(root / "1.npz", 70, label=0, seed=1)
    _write_recording(root / "2.npz", 70, label=1, seed=2)
    _write_recording(root / "3.npz", 10, label=1, seed=3)  # too short: skipped
    return root


def test_dataset_windows(data_dir: Path) -> None:
    ds = CalibrationDataset(data_dir, n_cols=32, stride=32, normalise=True)
    assert len(ds) == 4  # 2 windows (start 0 and 32) per long recording
    x, y = ds[0]
    assert x.shape == (1, N_FFT, 32) and x.dtype == torch.float32
    assert abs(float(x.mean())) < 1e-5 and abs(float(x.std(unbiased=False)) - 1.0) < 1e-4
    assert sorted(int(ds[i][1]) for i in range(len(ds))) == [0, 0, 1, 1]
    assert ds.n_fft == N_FFT


def test_dataset_window_content_is_transposed(data_dir: Path) -> None:
    ds = CalibrationDataset(data_dir, n_cols=32, stride=16, normalise=False)
    with np.load(data_dir / "1.npz") as z:
        spec = z["spectrogram_db"].astype(np.float32)
    x, _ = ds[1]  # second window of the first file: columns 16..47
    np.testing.assert_array_equal(x[0].numpy(), spec[16:48].T)


def test_model_shapes() -> None:
    model = SpectrogramAutoencoder.from_config({"model": SMALL_MODEL})
    x = torch.randn(3, 1, N_FFT, 32)
    x_hat, logits = model(x)
    assert x_hat.shape == x.shape
    assert logits.shape == (3, 1)
    assert model.latent_shape == (8, N_FFT // 4, 8)
    with pytest.raises(ValueError):
        model(torch.randn(3, 1, N_FFT + 1, 32))


def test_training_runs_end_to_end(data_dir: Path, tmp_path: Path) -> None:
    results = tmp_path / "results"
    cfg = {
        "seed": 0,
        "device": "cpu",
        "data": {
            "data_root": str(data_dir.parent), "train_subdir": "train", "val_subdir": "val",
            "val_split": 0.25, "n_cols": 32, "stride": 32, "normalise": True,
        },
        "dataloader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
        "model": SMALL_MODEL,
        "training": {
            "epochs": 2,
            "loss": {"reconstruction": "mse", "alpha": 0.5, "bce_pos_weight": None},
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999]},
            "scheduler": {"name": "steplr", "step_size": 1, "gamma": 0.5},
            "best_metric": "val_total",
            "progress_interval": 1,
        },
        "output": {
            "results_dir": str(results), "checkpoint_name": "best.pt",
            "log_file": str(results / "train.log"), "save_metrics": True,
            "metrics_file": str(results / "history.csv"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    load_script("train").main(["--config", str(config_path)])

    checkpoint = torch.load(results / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["model"] == SMALL_MODEL
    history = (results / "history.csv").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1 + 2  # header + one line per epoch
    assert (results / "train.log").is_file()
