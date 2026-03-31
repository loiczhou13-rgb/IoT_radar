from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RadioConfig:
    fc_hz: float = 2.4e9
    fs_hz: float = 600_000  # keep stable; Pluto hack supports higher but prefer conservative defaults
    rx_buffer_size: int = 65_536

    tx_gain_db: int = -40
    rx_gain_db: int = 55
    rx_gain_mode: str = "manual"  # "manual" | "slow_attack"

    pluto_uri: str = "ip:192.168.2.1"


@dataclass(frozen=True)
class TxConfig:
    waveform: str = "cw_tone"  # cw_tone | noise | qpsk | chirp
    amplitude: float = 0.5
    tone_hz: float = 10_000.0


@dataclass(frozen=True)
class StftConfig:
    window_size: int = 2048
    hop_size: int = 256  # ~87.5% overlap
    nfft: int = 4096
    window: str = "hann"  # hann | hamming

    history_frames: int = 220  # number of spectrogram time slices to keep
    update_ms: int = 80


@dataclass(frozen=True)
class ClutterConfig:
    method: str = "iir"  # mean | iir | mti
    iir_alpha: float = 0.98


@dataclass(frozen=True)
class RespirationTarget:
    min_hz: float = 0.10  # 6 bpm
    max_hz: float = 0.50  # 30 bpm
    min_stable_seconds: float = 6.0


@dataclass(frozen=True)
class DetectorConfig:
    velocity_band_min_ms: float = 0.01
    velocity_band_max_ms: float = 0.40
    snr_db_min: float = 6.0
    confidence_min: float = 0.6


@dataclass(frozen=True)
class AppConfig:
    radio: RadioConfig = RadioConfig()
    tx: TxConfig = TxConfig()
    stft: StftConfig = StftConfig()
    clutter: ClutterConfig = ClutterConfig()
    target: RespirationTarget = RespirationTarget()
    detector: DetectorConfig = DetectorConfig()

