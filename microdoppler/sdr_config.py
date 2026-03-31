from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PLUTO_HACK_BW_MAX_HZ = 56_000_000


def _fmt_hz(x: float) -> str:
    if x >= 1e6:
        return f"{x/1e6:.3g} MHz"
    if x >= 1e3:
        return f"{x/1e3:.3g} kHz"
    return f"{x:.3g} Hz"


@dataclass(frozen=True)
class PlutoSdrConfig:
    uri: str = "ip:192.168.2.1"
    fc_hz: float = 2.4e9
    fs_hz: float = 600_000
    rx_buffer_size: int = 65_536

    rx_gain_mode: str = "manual"  # manual | slow_attack
    rx_gain_db: int = 55
    tx_gain_db: int = -40

    tx_cyclic: bool = True

    def validate(self) -> list[str]:
        warnings: list[str] = []

        if self.fs_hz <= 0:
            raise ValueError("fs_hz must be > 0")

        # Pluto instantaneous BW is tied to sample_rate; keep a simple guardrail.
        if self.fs_hz > PLUTO_HACK_BW_MAX_HZ:
            warnings.append(
                "PlutoSDR limitation exceeded: requested sample_rate "
                f"{_fmt_hz(self.fs_hz)} > BW max {_fmt_hz(PLUTO_HACK_BW_MAX_HZ)}. "
                "Driver may clamp or fail; reduce fs_hz."
            )

        if self.rx_buffer_size <= 0:
            raise ValueError("rx_buffer_size must be > 0")

        if self.rx_gain_mode not in ("manual", "slow_attack"):
            warnings.append(
                f"Unknown rx_gain_mode={self.rx_gain_mode!r}; expected 'manual' or 'slow_attack'."
            )

        return warnings


def print_warnings(warnings: Iterable[str]) -> None:
    for w in warnings:
        print(f"[WARN] {w}")

