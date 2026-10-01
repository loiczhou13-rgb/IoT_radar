"""Regenerate ``tests/data/references.npz`` from the current code.

Run from the repository root::

    python tests/make_references.py

Only when a change of the numerical results is intended, in a dedicated
commit that explains why (during the refactor: only the fixes of bugs B2 and
B5, see ``REFACTOR_PLAN.md``).  Any other change of these references hides a
numerical regression.
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
