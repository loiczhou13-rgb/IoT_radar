from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ClutterFilterState:
    iir_mean: Optional[np.ndarray] = None
    prev_frame: Optional[np.ndarray] = None


def remove_dc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    return x - np.mean(x)


def apply_clutter_filter(
    x: np.ndarray,
    state: ClutterFilterState,
    method: str = "iir",
    iir_alpha: float = 0.98,
) -> np.ndarray:
    """
    Apply clutter suppression BEFORE STFT, following the PDF guidance.
    - mean: subtract running mean over frames (clutter fixe)
    - iir:  1st order IIR mean (clutter variable), alpha in [0.9, 0.999]
    - mti:  x[n] - x[n-1]
    """
    x = remove_dc(x)

    if method == "mean":
        if state.iir_mean is None:
            state.iir_mean = x.copy()
        else:
            state.iir_mean = state.iir_mean + x
        mean = state.iir_mean / 2.0 if state.iir_mean is not None else 0.0
        return x - mean

    if method == "iir":
        a = float(np.clip(iir_alpha, 0.0, 0.999999))
        if state.iir_mean is None:
            state.iir_mean = x.copy()
        else:
            state.iir_mean = a * state.iir_mean + (1.0 - a) * x
        return x - state.iir_mean

    if method == "mti":
        if state.prev_frame is None:
            y = x
        else:
            y = x - state.prev_frame
        state.prev_frame = x.copy()
        return y

    raise ValueError(f"Unknown clutter method: {method!r}")

