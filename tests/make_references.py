"""Regenerate ``tests/data/references.npz``: ``python tests/make_references.py``.

Only for an intended change of the numerical results, in a commit that says why.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from characterization_cases import CASES

REFERENCE_FILE = Path(__file__).resolve().parent / "data" / "references.npz"


def main() -> None:
    """Run every characterization case and store its outputs."""
    logging.disable(logging.WARNING)
    arrays: dict[str, np.ndarray] = {}
    for name, case in CASES.items():
        for key, value in case().items():
            arrays[f"{name}/{key}"] = np.asarray(value)
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REFERENCE_FILE, **arrays)
    print(f"{len(arrays)} reference arrays written to {REFERENCE_FILE}")


if __name__ == "__main__":
    main()
