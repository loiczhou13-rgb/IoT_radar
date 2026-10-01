"""Supervised micro-Doppler autoencoder training (library part).

No hyperparameters are hard-coded here: everything is read from the training
configuration (``configs/training.yaml`` by default).  The command-line
entry point is ``scripts/train.py``, whose options (``--epochs``…) override
the YAML values for quick experiments.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from iot_radar.config import resolve_repo_path, setup_logging
from iot_radar.ml.dataset import CalibrationDataset
from iot_radar.ml.model import SpectrogramAutoencoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / setup helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(spec: str) -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a simple ASCII table to stdout."""
    if not rows:
        return
    key_w = max(len(k) for k, _ in rows)
    val_w = max(len(v) for _, v in rows)
    rule = f"+-{'-' * key_w}-+-{'-' * val_w}-+"
    print()
    print(title)
    print(rule)
    for key, val in rows:
        print(f"| {key:<{key_w}} | {val:<{val_w}} |")
    print(rule)


# ---------------------------------------------------------------------------
# Metrics history (CSV)
# ---------------------------------------------------------------------------

_METRICS_FIELDS = (
    "epoch", "lr",
    "train_total", "train_recon", "train_bce", "train_acc",
    "val_total", "val_recon", "val_bce", "val_acc",
)


class MetricsHistory:
    """Append-only CSV writer for per-epoch train/val metrics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=_METRICS_FIELDS).writeheader()

    def append(
        self,
        epoch: int,
        lr: float,
        train_m: dict[str, float],
        val_m: dict[str, float],
    ) -> None:
        row = {
            "epoch": epoch,
            "lr": f"{lr:.6e}",
            "train_total": f"{train_m['total']:.6f}",
            "train_recon": f"{train_m['recon']:.6f}",
            "train_bce": f"{train_m['bce']:.6f}",
            "train_acc": f"{train_m['acc']:.6f}",
            "val_total": f"{val_m['total']:.6f}",
            "val_recon": f"{val_m['recon']:.6f}",
            "val_bce": f"{val_m['bce']:.6f}",
            "val_acc": f"{val_m['acc']:.6f}",
        }
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_METRICS_FIELDS).writerow(row)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _has_npz(directory: Path) -> bool:
    return directory.is_dir() and any(directory.rglob("*.npz"))


def build_loaders(
    cfg: dict[str, Any],
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Build train / val DataLoaders from the config.

    - If ``data.val_subdir`` exists and contains ``.npz`` files, it is used
      for validation.
    - Otherwise, ``data.train_subdir`` is split randomly according to
      ``data.val_split``.

    Returns loaders and a metadata dict (sample counts, split description).
    """
    data_cfg = cfg["data"]
    dl_cfg = cfg["dataloader"]

    data_root = resolve_repo_path(data_cfg["data_root"])
    train_dir = data_root / data_cfg["train_subdir"]
    val_dir = data_root / data_cfg["val_subdir"]

    ds_kwargs = dict(
        n_cols=int(data_cfg["n_cols"]),
        stride=int(data_cfg["stride"]),
        normalise=bool(data_cfg["normalise"]),
    )

    if _has_npz(val_dir):
        train_ds: torch.utils.data.Dataset = CalibrationDataset(train_dir, **ds_kwargs)
        val_ds: torch.utils.data.Dataset = CalibrationDataset(val_dir, **ds_kwargs)
        split_desc = f"train={train_dir.name}/  val={val_dir.name}/"
        logger.info("Validation depuis %s — train depuis %s", val_dir, train_dir)
    else:
        full = CalibrationDataset(train_dir, **ds_kwargs)
        val_split = float(data_cfg["val_split"])
        if not 0.0 < val_split < 1.0:
            raise ValueError("data.val_split doit être dans ]0, 1[.")
        n_total = len(full)
        n_val = max(1, int(round(n_total * val_split)))
        n_train = n_total - n_val
        if n_train <= 0:
            raise ValueError(
                f"Pas assez de fenêtres ({n_total}) pour split {val_split}."
            )
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=generator)
        split_desc = f"auto split {1 - val_split:.0%}/{val_split:.0%} from {train_dir.name}/"
        logger.info(
            "Split auto depuis %s — train=%d / val=%d (val_split=%.2f)",
            train_dir, n_train, n_val, val_split,
        )

    common = dict(
        batch_size=int(dl_cfg["batch_size"]),
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg["pin_memory"]),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    meta = {
        "split": split_desc,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "batch_size": int(dl_cfg["batch_size"]),
    }
    return train_loader, val_loader, meta


# ---------------------------------------------------------------------------
# Loss / optim / scheduler
# ---------------------------------------------------------------------------

def build_recon_loss(name: str) -> nn.Module:
    name = (name or "mse").lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"loss.reconstruction inconnu : {name!r} (attendu mse|l1)")


def build_bce_loss(pos_weight: float | None) -> nn.BCEWithLogitsLoss:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32)
    )


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = cfg["training"]["optimizer"]
    name = (opt_cfg.get("name") or "adamw").lower()
    if name != "adamw":
        raise ValueError(f"optimizer.name non supporté : {name!r} (attendu adamw)")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(float(b) for b in opt_cfg.get("betas", (0.9, 0.999))),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> torch.optim.lr_scheduler._LRScheduler:
    sch_cfg = cfg["training"]["scheduler"]
    name = (sch_cfg.get("name") or "steplr").lower()
    if name != "steplr":
        raise ValueError(f"scheduler.name non supporté : {name!r} (attendu steplr)")
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(sch_cfg["step_size"]),
        gamma=float(sch_cfg["gamma"]),
    )


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def _run_epoch(
    *,
    model: SpectrogramAutoencoder,
    loader: Iterable,
    device: torch.device,
    recon_loss_fn: nn.Module,
    bce_loss_fn: nn.BCEWithLogitsLoss,
    alpha: float,
    optimizer: torch.optim.Optimizer | None,
    show_progress: bool = False,
    epoch: int | None = None,
    n_epochs: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running = {"total": 0.0, "recon": 0.0, "bce": 0.0}
    n_samples = 0
    n_correct = 0

    batch_iter: Iterable = loader
    pbar: tqdm | None = None
    if show_progress:
        desc = f"Epoch {epoch}/{n_epochs}" if epoch is not None else "Train"
        pbar = tqdm(loader, desc=desc, leave=True)
        batch_iter = pbar

    for x, y in batch_iter:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            x_hat, logits = model(x)
            logits = logits.squeeze(1)               # (B,)
            y_float = y.float()                      # (B,) for BCE

            recon = recon_loss_fn(x_hat, x)
            bce = bce_loss_fn(logits, y_float)
            loss = alpha * recon + (1.0 - alpha) * bce

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = x.shape[0]
        running["total"] += loss.item() * bs
        running["recon"] += recon.item() * bs
        running["bce"] += bce.item() * bs
        n_samples += bs
        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.5).long()
            n_correct += int((pred == y).sum().item())

        if pbar is not None:
            pbar.set_postfix(
                loss=f"{running['total'] / n_samples:.4f}",
                acc=f"{n_correct / n_samples:.3f}",
                refresh=False,
            )

    if pbar is not None:
        pbar.set_postfix(
            loss=f"{running['total'] / max(n_samples, 1):.4f}",
            acc=f"{n_correct / max(n_samples, 1):.3f}",
        )
        pbar.close()

    return {
        "total": running["total"] / max(n_samples, 1),
        "recon": running["recon"] / max(n_samples, 1),
        "bce": running["bce"] / max(n_samples, 1),
        "acc": n_correct / max(n_samples, 1),
    }


def _should_display_epoch(epoch: int, n_epochs: int, interval: int) -> bool:
    if interval <= 1:
        return True
    return epoch % interval == 0 or epoch == n_epochs


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

def train(cfg: dict[str, Any], config_path: Path) -> None:
    """Train the autoencoder described by *cfg* and save the best checkpoint.

    *config_path* is only displayed in the run summary.
    """
    seed = int(cfg.get("seed", 0))
    _set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    out_cfg = cfg["output"]
    results_dir = resolve_repo_path(out_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = resolve_repo_path(out_cfg["log_file"]) if out_cfg.get("log_file") else None
    # Console kept quiet (progress bar + prints); every INFO record goes to the file.
    setup_logging("INFO", log_file, console_level="WARNING", file_mode="a")

    device = _pick_device(cfg.get("device", "auto"))
    logging.getLogger("iot_radar.ml.dataset").setLevel(logging.WARNING)

    print("Loading data...", flush=True)
    train_loader, val_loader, data_meta = build_loaders(cfg, generator=generator)
    print("Loading complete.", flush=True)

    model = SpectrogramAutoencoder.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loss_cfg = cfg["training"]["loss"]
    alpha = float(loss_cfg["alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("training.loss.alpha doit être dans [0, 1].")
    recon_loss_fn = build_recon_loss(loss_cfg["reconstruction"]).to(device)
    bce_loss_fn = build_bce_loss(loss_cfg.get("bce_pos_weight")).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    best_metric_key = cfg["training"].get("best_metric", "val_total")
    if best_metric_key not in ("val_total", "val_bce", "val_recon"):
        raise ValueError(
            f"training.best_metric doit être val_total|val_bce|val_recon, "
            f"reçu {best_metric_key!r}."
        )
    metric_field = best_metric_key.split("_", 1)[1]

    ckpt_path = results_dir / out_cfg["checkpoint_name"]
    best_value = float("inf")
    n_epochs = int(cfg["training"]["epochs"])
    progress_interval = max(1, int(cfg["training"].get("progress_interval", 10)))

    save_metrics = bool(out_cfg.get("save_metrics", False))
    metrics_history: MetricsHistory | None = None
    metrics_path: Path | None = None
    if save_metrics:
        metrics_path = resolve_repo_path(out_cfg["metrics_file"])
        metrics_history = MetricsHistory(metrics_path)

    opt_cfg = cfg["training"]["optimizer"]
    sch_cfg = cfg["training"]["scheduler"]
    recon_name = loss_cfg["reconstruction"].upper()

    _print_table("Run configuration", [
        ("Device", str(device)),
        ("Seed", str(seed)),
        ("Config", str(config_path)),
        ("Data split", data_meta["split"]),
        ("Train windows", str(data_meta["train_samples"])),
        ("Val windows", str(data_meta["val_samples"])),
        ("Batch size", str(data_meta["batch_size"])),
        ("Train batches", str(data_meta["train_batches"])),
        ("Val batches", str(data_meta["val_batches"])),
        ("Model params", f"{n_params:,}"),
        ("Input shape", str(model.expected_input_shape)),
        ("Latent shape", str(model.latent_shape)),
        ("Loss", f"{alpha:.2f}·{recon_name} + {1 - alpha:.2f}·BCE"),
        ("Optimizer", f"AdamW  lr={opt_cfg['learning_rate']}  wd={opt_cfg['weight_decay']}"),
        ("Scheduler", f"StepLR  step={sch_cfg['step_size']}  γ={sch_cfg['gamma']}"),
        ("Epochs", str(n_epochs)),
        ("Best metric", best_metric_key),
        ("Progress bar", f"every {progress_interval} epoch(s)"),
        ("Metrics CSV", str(metrics_path) if metrics_path else "disabled"),
        ("Checkpoint", str(ckpt_path)),
    ])

    logger.info(
        "Device=%s | train=%d val=%d windows | batch=%d",
        device, data_meta["train_samples"], data_meta["val_samples"],
        data_meta["batch_size"],
    )
    logger.info(
        "Model: %s params | input=%s latent=%s",
        f"{n_params:,}", model.expected_input_shape, model.latent_shape,
    )
    if metrics_path is not None:
        logger.info("Metrics CSV → %s", metrics_path)

    print("\nTraining started.", flush=True)

    for epoch in range(1, n_epochs + 1):
        show = _should_display_epoch(epoch, n_epochs, progress_interval)

        train_m = _run_epoch(
            model=model, loader=train_loader, device=device,
            recon_loss_fn=recon_loss_fn, bce_loss_fn=bce_loss_fn,
            alpha=alpha, optimizer=optimizer,
            show_progress=show,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        val_m = _run_epoch(
            model=model, loader=val_loader, device=device,
            recon_loss_fn=recon_loss_fn, bce_loss_fn=bce_loss_fn,
            alpha=alpha, optimizer=None,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if metrics_history is not None:
            metrics_history.append(epoch, current_lr, train_m, val_m)

        if show:
            print(
                f"Epoch {epoch}/{n_epochs}  val  "
                f"loss={val_m['total']:.4f}  acc={val_m['acc']:.3f}"
            )

        current = val_m[metric_field]
        if current < best_value:
            best_value = current
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg,
                    "best_metric": best_metric_key,
                    "best_value": best_value,
                    "val_metrics": val_m,
                },
                ckpt_path,
            )
            if show:
                print(f"  ↳ checkpoint ({best_metric_key}={best_value:.4f})")
            logger.info(
                "Checkpoint saved (%s=%.4f) → %s",
                best_metric_key, best_value, ckpt_path,
            )

    print(f"\nTraining finished. Best {best_metric_key} = {best_value:.4f} → {ckpt_path}")
    logger.info("Training finished. Best %s = %.4f", best_metric_key, best_value)
