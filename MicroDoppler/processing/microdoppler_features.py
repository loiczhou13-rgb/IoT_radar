from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MicroDopplerFeatures:
    band_energy: float
    band_snr_db: float
    centroid_v_ms: float


def _db(x: float) -> float:
    return 10.0 * float(np.log10(max(x, 1e-12)))


def extract_features(
    *,
    v_ms: np.ndarray,
    psd_db: np.ndarray,
    band_v_min_ms: float,
    band_v_max_ms: float,
) -> MicroDopplerFeatures:
    """
    psd_db: 2D array (freq, time). We use the most recent time slice for features.
    """
    v_ms = np.asarray(v_ms, dtype=np.float32)
    last = np.asarray(psd_db[:, -1], dtype=np.float32)

    mask = (np.abs(v_ms) >= band_v_min_ms) & (np.abs(v_ms) <= band_v_max_ms)
    if not np.any(mask):
        return MicroDopplerFeatures(band_energy=0.0, band_snr_db=-999.0, centroid_v_ms=0.0)

    # Convert to linear power
    p = 10 ** (last / 10.0)
    p_band = p[mask]
    v_band = v_ms[mask]

    band_energy = float(np.mean(p_band))
    noise = float(np.median(p))
    band_snr_db = _db(band_energy / max(noise, 1e-12))

    centroid_v = float(np.sum(v_band * p_band) / max(np.sum(p_band), 1e-12))
    return MicroDopplerFeatures(band_energy=band_energy, band_snr_db=band_snr_db, centroid_v_ms=centroid_v)

