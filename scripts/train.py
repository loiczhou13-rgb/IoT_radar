#!/usr/bin/env python3
"""Supervised micro-Doppler autoencoder training.

No hyperparameters are hard-coded: everything is read from the training
configuration (``configs/training.yaml`` by default, or another YAML
passed via ``--config``).

Run (from the repository root) ::

    python scripts/train.py
    python scripts/train.py --config my_training.yaml --epochs 100

The ``--epochs`` option (and a few others) **overrides** the YAML value
for quick experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from iot_radar.config import DEFAULT_TRAINING_CONFIG, load_config
from iot_radar.ml.train import train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (*argv* defaults to ``sys.argv``)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="YAML file (default: configs/training.yaml)",
    )
    p.add_argument("--epochs", type=int, default=None,
                   help="Overrides training.epochs.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Overrides dataloader.batch_size.")
    p.add_argument("--lr", type=float, default=None,
                   help="Overrides training.optimizer.learning_rate.")
    p.add_argument("--device", default=None,
                   help="Overrides the device (auto|cpu|cuda).")
    return p.parse_args(argv)


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Replace the configuration values given on the command line."""
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["dataloader"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["training"]["optimizer"]["learning_rate"] = float(args.lr)
    if args.device is not None:
        cfg["device"] = str(args.device)


def main(argv: list[str] | None = None) -> None:
    """Load the configuration, apply the overrides and train."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)
    train(cfg, args.config)


if __name__ == "__main__":
    main()
