from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sdr_config import PlutoSdrConfig, print_warnings

import json
import os
import time


_DEBUG_LOG_PATH = "/home/zhoul/IoT_radar/.cursor/debug-dfe351.log"


def _dlog(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "dfe351",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


try:
    import adi  # type: ignore
except Exception:  # pragma: no cover
    adi = None


@dataclass
class SdrStatus:
    applied_fc_hz: Optional[float] = None
    applied_fs_hz: Optional[float] = None
    applied_rx_buffer_size: Optional[int] = None


class PlutoSdrIO:
    def __init__(self, cfg: PlutoSdrConfig):
        warnings = cfg.validate()
        print_warnings(warnings)

        if adi is None:
            raise RuntimeError(
                "pyadi-iio is not available. Install it (pip install pyadi-iio) "
                "and ensure libiio is available for PlutoSDR."
            )

        self.cfg = cfg
        # #region agent log
        _dlog(
            run_id="pre-fix",
            hypothesis_id="H1_uri_or_transport",
            location="microdoppler/sdr_io.py:PlutoSdrIO.__init__",
            message="Attempting adi.Pluto connection",
            data={
                "uri": cfg.uri,
                "fc_hz": cfg.fc_hz,
                "fs_hz": cfg.fs_hz,
                "rx_buffer_size": cfg.rx_buffer_size,
                "DISPLAY": os.environ.get("DISPLAY"),
                "WSL_DISTRO_NAME": os.environ.get("WSL_DISTRO_NAME"),
            },
        )
        # #endregion
        try:
            self.sdr = adi.Pluto(cfg.uri)
        except Exception as e:
            # #region agent log
            _dlog(
                run_id="pre-fix",
                hypothesis_id="H2_no_device_found",
                location="microdoppler/sdr_io.py:PlutoSdrIO.__init__",
                message="adi.Pluto connection failed",
                data={"uri": cfg.uri, "error_type": type(e).__name__, "error": str(e)},
            )
            # #endregion
            raise
        self.status = SdrStatus()
        self._apply_config()

    def _apply_config(self) -> None:
        self.sdr.sample_rate = int(self.cfg.fs_hz)
        self.sdr.rx_lo = int(self.cfg.fc_hz)
        self.sdr.tx_lo = int(self.cfg.fc_hz)
        self.sdr.rx_buffer_size = int(self.cfg.rx_buffer_size)

        self.sdr.rx_gain_control_mode_chan0 = self.cfg.rx_gain_mode
        if self.cfg.rx_gain_mode == "manual":
            self.sdr.rx_hardwaregain_chan0 = int(self.cfg.rx_gain_db)

        self.sdr.tx_gain_control_mode_chan0 = "manual"
        self.sdr.tx_hardwaregain_chan0 = int(self.cfg.tx_gain_db)

        # Applied values (read-back when possible)
        self.status.applied_fc_hz = float(getattr(self.sdr, "rx_lo", self.cfg.fc_hz))
        self.status.applied_fs_hz = float(getattr(self.sdr, "sample_rate", self.cfg.fs_hz))
        self.status.applied_rx_buffer_size = int(getattr(self.sdr, "rx_buffer_size", self.cfg.rx_buffer_size))

        if abs(self.status.applied_fs_hz - self.cfg.fs_hz) > 1:
            print(
                "[WARN] PlutoSDR clamped sample_rate: requested "
                f"{self.cfg.fs_hz} Hz, applied {self.status.applied_fs_hz} Hz."
            )

    def rx(self) -> np.ndarray:
        # Pluto I/Q is typically int16 scaled; pyadi-iio returns complex64 already.
        x = self.sdr.rx()
        x = np.asarray(x, dtype=np.complex64)
        return x / (2**11)

    def tx(self, iq: np.ndarray, cyclic: Optional[bool] = None) -> None:
        if cyclic is None:
            cyclic = self.cfg.tx_cyclic

        self.sdr.tx_destroy_buffer()
        self.sdr.tx_cyclic_buffer = bool(cyclic)
        iq = np.asarray(iq, dtype=np.complex64)
        self.sdr.tx(iq * (2**11))

    def stop_tx(self) -> None:
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass

