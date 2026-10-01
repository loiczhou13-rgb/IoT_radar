"""Headless smoke test of the pygame home screen (SDL dummy video driver)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from iot_radar.ui.launcher import HomeScreen  # noqa: E402
from script_loader import load_script  # noqa: E402


@pytest.mark.filterwarnings("ignore:no fast renderer available")
def test_buttons() -> None:
    clicked: list[str] = []
    screen = HomeScreen([("first", lambda: clicked.append("first")), ("second", lambda: clicked.append("second"))])
    screen.draw((0, 0))  # places the buttons

    click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)
    assert screen.handle_event(click, screen.btn_theme.center)
    assert screen.theme_mode == "dark"
    assert screen.handle_event(click, screen.action_buttons[1].center)
    assert clicked == ["second"] and screen.pressed_action == 1
    assert screen.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1), (0, 0))
    assert screen.pressed_action is None
    assert not screen.handle_event(click, screen.btn_quit.center)
    pygame.quit()


def test_launcher_runs_the_radar_in_a_subprocess(monkeypatch) -> None:
    """Bug B4: the radar must not run in a thread of the pygame process."""
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, cwd: calls.append(args))
    launcher = load_script("launcher")
    assert [label for label, _ in launcher.ACTIONS] == ["Radar — PlutoSDR", "Radar — simulation"]
    for _, action in launcher.ACTIONS:
        action()
    assert calls[0][0] == sys.executable and calls[0][1].endswith("scripts/run_radar.py")
    assert calls[0][2:] == [] and calls[1][2:] == ["--simulation"]
