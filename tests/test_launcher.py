"""Headless smoke test of the pygame home screen (SDL dummy video driver)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from iot_radar.ui.launcher import HomeScreen  # noqa: E402


@pytest.mark.filterwarnings("ignore:no fast renderer available")
def test_buttons() -> None:
    started: list[bool] = []
    screen = HomeScreen(start_acquisition=lambda: started.append(True))
    screen.draw((0, 0))  # places the buttons

    click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)
    assert screen.handle_event(click, screen.btn_theme.center)
    assert screen.theme_mode == "dark"
    assert screen.handle_event(click, screen.btn_acquisition.center)
    assert started == [True] and screen.acquisition_pressed
    assert screen.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1), (0, 0))
    assert not screen.acquisition_pressed
    assert not screen.handle_event(click, screen.btn_quit.center)
    pygame.quit()


def test_launcher_script_runs_the_radar_in_a_subprocess(monkeypatch) -> None:
    """Bug B4: the radar must not run in a thread of the pygame process."""
    import subprocess
    import sys

    from script_loader import load_script

    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, cwd: calls.append((args, cwd)))
    launcher = load_script("launcher")
    launcher.start_radar()
    (args, cwd), = calls
    assert args[0] == sys.executable and args[1].endswith("scripts/run_radar.py")
