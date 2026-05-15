#!/usr/bin/env python3
"""Enchaîne plusieurs acquisitions ``record_acquisition`` avec la même configuration.

Exemple — 5 échantillons train, salle, label 1, 60 s chacun, 2 min de pause entre deux ::

    cd MicroDopplerDetection
    python utils/auto_record.py -n 5 --interval 120 \\
        --subset train --env salle --label 1 --duration 60

``--interval`` : secondes d’attente **après la fin** d’un enregistrement avant
de lancer le suivant (0 par défaut).

Les indices d’échantillon sont toujours auto (« prochain libre ») : ne pas
passer ``--index``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent
_ROOT = _UTILS_DIR.parent


def _ensure_root_on_path() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def main() -> None:
    _ensure_root_on_path()
    from utils.record_acquisition import build_record_argument_parser, run as record_run

    p = build_record_argument_parser(
        description=(
            "Enchaîner N enregistrements micro-Doppler (mêmes options que "
            "record_acquisition, plus -n / --interval)."
        ),
    )
    g = p.add_argument_group("enchaînement")
    g.add_argument(
        "--samples",
        "-n",
        type=int,
        required=True,
        metavar="N",
        help="Nombre d’échantillons à enregistrer successivement.",
    )
    g.add_argument(
        "--interval",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "Pause en secondes après la fin d’un .npz avant le suivant "
            "(défaut : 0)."
        ),
    )

    args = p.parse_args()
    if args.samples < 1:
        raise SystemExit("-n / --samples doit être >= 1.")
    if args.interval < 0:
        raise SystemExit("--interval doit être >= 0.")
    if args.index is not None:
        raise SystemExit(
            "--index est incompatible avec auto_record (indices auto : prochain libre).",
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    paths: list[Path] = []
    for k in range(args.samples):
        if k > 0 and args.interval > 0:
            log.info("Pause %.1f s avant l’échantillon %d / %d", args.interval, k + 1, args.samples)
            time.sleep(args.interval)
        log.info(
            "Début échantillon %d / %d (subset=%s env=%s label=%s)",
            k + 1,
            args.samples,
            args.subset,
            args.env,
            args.label,
        )
        paths.append(record_run(args))

    log.info("Terminé — %d fichier(s) .npz : %s", len(paths), ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
