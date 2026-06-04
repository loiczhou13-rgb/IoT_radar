#!/usr/bin/env python3
"""Entraînement de l'autoencodeur supervisé micro-Doppler.

Aucun hyperparamètre n'est codé en dur dans ce script : tout est lu dans
``AICalibration/config.yaml`` (ou un autre YAML passé via ``--config``).

Lance ::

    cd AICalibration                   # ou depuis la racine du dépôt
    python train.py
    python train.py --config config.yaml --epochs 100

L'option ``--epochs`` (et quelques autres) **surcharge** la valeur YAML
pour des essais rapides.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, random_split

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from AICalibration.dataset import CalibrationDataset  # noqa: E402
from AICalibration.model import SpectrogramAutoencoder  # noqa: E402

logger = logging.getLogger("AICalibration.train")


# ---------------------------------------------------------------------------
# Config / setup helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config invalide (pas un mapping) : {path}")
    return cfg


def _resolve_repo_path(p: str | Path) -> Path:
    """Chemin relatif → résolu par rapport à la racine du dépôt."""
    pth = Path(p).expanduser()
    if pth.is_absolute():
        return pth.resolve()
    return (_REPO_ROOT / pth).resolve()


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


def _setup_logging(log_file: Path | None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(fh)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _has_npz(directory: Path) -> bool:
    return directory.is_dir() and any(directory.rglob("*.npz"))


def build_loaders(
    cfg: dict[str, Any],
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader]:
    """Crée les DataLoaders train / val à partir de la config.

    - Si ``data.val_subdir`` existe et contient des ``.npz``, il sert de
      validation.
    - Sinon, on découpe aléatoirement ``data.train_subdir`` selon
      ``data.val_split``.
    """
    data_cfg = cfg["data"]
    dl_cfg = cfg["dataloader"]

    data_root = _resolve_repo_path(data_cfg["data_root"])
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
    return train_loader, val_loader


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
        lr=float(opt_cfg["lr"]),
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
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running = {"total": 0.0, "recon": 0.0, "bce": 0.0}
    n_samples = 0
    n_correct = 0

    for x, y in loader:
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

    return {
        "total": running["total"] / max(n_samples, 1),
        "recon": running["recon"] / max(n_samples, 1),
        "bce": running["bce"] / max(n_samples, 1),
        "acc": n_correct / max(n_samples, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--config",
        type=Path,
        default=_THIS_DIR / "config.yaml",
        help="Fichier YAML (défaut : AICalibration/config.yaml)",
    )
    p.add_argument("--epochs", type=int, default=None,
                   help="Surcharge training.epochs.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Surcharge dataloader.batch_size.")
    p.add_argument("--lr", type=float, default=None,
                   help="Surcharge training.optimizer.lr.")
    p.add_argument("--device", default=None,
                   help="Surcharge le device (auto|cpu|cuda).")
    return p.parse_args()


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["dataloader"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["training"]["optimizer"]["lr"] = float(args.lr)
    if args.device is not None:
        cfg["device"] = str(args.device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    cfg = _load_yaml(args.config)
    _apply_cli_overrides(cfg, args)

    seed = int(cfg.get("seed", 0))
    _set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    out_cfg = cfg["output"]
    results_dir = _resolve_repo_path(out_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = _resolve_repo_path(out_cfg["log_file"]) if out_cfg.get("log_file") else None
    _setup_logging(log_file)

    device = _pick_device(cfg.get("device", "auto"))
    logger.info("Device sélectionné : %s", device)
    logger.info("Config : %s", args.config)
    logger.info("Seed   : %d", seed)

    # Data
    train_loader, val_loader = build_loaders(cfg, generator=generator)
    logger.info(
        "Loaders prêts — train batches: %d, val batches: %d, batch_size=%d",
        len(train_loader), len(val_loader),
        int(cfg["dataloader"]["batch_size"]),
    )

    # Model
    model = SpectrogramAutoencoder.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Modèle prêt — paramètres=%s, expected_input=%s, latent=%s",
        f"{n_params:,}",
        model.expected_input_shape,
        model.latent_shape,
    )

    # Loss / optim / sched
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

    logger.info(
        "Loss : alpha·%s + (1-alpha)·BCE avec alpha=%.2f, recon=%s",
        loss_cfg["reconstruction"].upper(),
        alpha,
        loss_cfg["reconstruction"].upper(),
    )

    for epoch in range(1, n_epochs + 1):
        train_m = _run_epoch(
            model=model, loader=train_loader, device=device,
            recon_loss_fn=recon_loss_fn, bce_loss_fn=bce_loss_fn,
            alpha=alpha, optimizer=optimizer,
        )
        val_m = _run_epoch(
            model=model, loader=val_loader, device=device,
            recon_loss_fn=recon_loss_fn, bce_loss_fn=bce_loss_fn,
            alpha=alpha, optimizer=None,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %02d/%02d  lr=%.2e  "
            "train: total=%.4f recon=%.4f bce=%.4f acc=%.3f  |  "
            "val: total=%.4f recon=%.4f bce=%.4f acc=%.3f",
            epoch, n_epochs, current_lr,
            train_m["total"], train_m["recon"], train_m["bce"], train_m["acc"],
            val_m["total"], val_m["recon"], val_m["bce"], val_m["acc"],
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
            logger.info(
                "↳ checkpoint sauvegardé (%s=%.4f) → %s",
                best_metric_key, best_value, ckpt_path,
            )

    logger.info("Entraînement terminé. Meilleur %s = %.4f", best_metric_key, best_value)
    logger.info("Checkpoint : %s", ckpt_path)


if __name__ == "__main__":
    main()
