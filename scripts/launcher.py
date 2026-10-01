#!/usr/bin/env python3
"""Home screen of the radar (pygame); its main button starts the live radar.

Run from the repository root::

    python scripts/launcher.py

The radar runs in its **own process** (``scripts/run_radar.py``): matplotlib's
Tk back-end must own the main thread of its process, and the radar parses its
own command line (bug B4: it used to run in a thread of the launcher).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from iot_radar.config import REPO_ROOT
from iot_radar.ui.launcher import HomeScreen

RUN_RADAR_SCRIPT: Path = Path(__file__).resolve().parent / "run_radar.py"


def start_radar() -> subprocess.Popen:
    """Start ``scripts/run_radar.py`` in a new Python process."""
    return subprocess.Popen([sys.executable, str(RUN_RADAR_SCRIPT)], cwd=REPO_ROOT)


if __name__ == "__main__":
    HomeScreen(start_acquisition=start_radar).run()
