#!/usr/bin/env python3
"""Home screen of the radar (pygame); its main button starts the live radar.

Run from the repository root::

    python scripts/launcher.py
"""

from __future__ import annotations

from run_radar import main as run_radar_main

from iot_radar.ui.launcher import HomeScreen

if __name__ == "__main__":
    HomeScreen(start_acquisition=run_radar_main).run()
