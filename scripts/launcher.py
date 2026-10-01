#!/usr/bin/env python3
"""Home screen of the radar (pygame): start the radar on the PlutoSDR or in simulation.

Run from the repository root::

    python scripts/launcher.py

Each button runs ``scripts/run_radar.py`` in its **own process**: matplotlib's
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


def start_radar(*options: str) -> subprocess.Popen:
    """Start ``scripts/run_radar.py`` with *options* in a new Python process."""
    return subprocess.Popen([sys.executable, str(RUN_RADAR_SCRIPT), *options], cwd=REPO_ROOT)


ACTIONS = [
    ("Radar — PlutoSDR", lambda: start_radar()),
    ("Radar — simulation", lambda: start_radar("--simulation")),
]
"""Buttons of the home screen: (label, function started on click)."""


if __name__ == "__main__":
    HomeScreen(ACTIONS).run()
