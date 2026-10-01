"""Bit-exact comparison of the DSP outputs with the stored references.

See ``tests/characterization_cases.py`` for the cases and the rules that
govern the reference file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from characterization_cases import CASES

REFERENCE_FILE = Path(__file__).resolve().parent / "data" / "references.npz"


@pytest.fixture(scope="module")
def references() -> dict[str, np.ndarray]:
    with np.load(REFERENCE_FILE) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_outputs_match_references(case_name: str, references: dict[str, np.ndarray]) -> None:
    logging.disable(logging.WARNING)
    try:
        outputs = CASES[case_name]()
    finally:
        logging.disable(logging.NOTSET)
    expected_keys = sorted(k.split("/", 1)[1] for k in references if k.startswith(f"{case_name}/"))
    assert sorted(outputs) == expected_keys
    for key, value in outputs.items():
        np.testing.assert_array_equal(
            np.asarray(value), references[f"{case_name}/{key}"],
            err_msg=f"{case_name}/{key} differs from the reference",
        )
