from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _window(name: str, n: int) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    raise ValueError(f"Unsupported window: {name!r}")


@dataclass(frozen=True)
class StftResult:
    fd_hz: np.ndarray
    v_ms: np.ndarray
    psd_db: np.ndarray  # shape (freq, time)
    dt_s: float
    df_hz: float
    dv_ms: float


def stft_microdoppler(
    x: np.ndarray,
    *,
    fs_hz: float,
    fc_hz: float,
    window_size: int,
    hop_size: int,
    nfft: int,
    window: str = "hann",
) -> StftResult:
    x = np.asarray(x, dtype=np.complex64)
    if len(x) < window_size:
        # pad to allow at least one frame
        x = np.pad(x, (0, window_size - len(x)))

    win = _window(window, window_size)
    win = win / (np.sqrt(np.mean(win**2)) + 1e-12)  # RMS normalize

    n_frames = 1 + max(0, (len(x) - window_size) // hop_size)
    frames = np.empty((n_frames, window_size), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_size
        frames[i] = x[start : start + window_size]

    frames = frames * win[None, :]
    spec = np.fft.fft(frames, n=nfft, axis=1)
    spec = np.fft.fftshift(spec, axes=1)

    mag = np.abs(spec).astype(np.float32)
    psd_db = 20.0 * np.log10(mag / (window_size + 1e-12) + 1e-12)

    fd_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs_hz)).astype(np.float32)
    c = 299_792_458.0
    lam = c / float(fc_hz)
    v_ms = (fd_hz * lam / 2.0).astype(np.float32)

    dt_s = float(hop_size / fs_hz)
    df_hz = float(fs_hz / nfft)
    dv_ms = float(df_hz * lam / 2.0)

    # transpose to (freq, time) for plotting convenience
    return StftResult(fd_hz=fd_hz, v_ms=v_ms, psd_db=psd_db.T, dt_s=dt_s, df_hz=df_hz, dv_ms=dv_ms)

