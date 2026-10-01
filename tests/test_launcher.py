"""Headless smoke test of the pygame home screen (SDL dummy video driver)."""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from iot_radar.ui.launcher import HomeScreen  # noqa: E402


@pytest.mark.filterwarnings("ignore:no fast renderer available")
def test_buttons() -> None:
    started = threading.Event()
    screen = HomeScreen(start_acquisition=started.set)
    screen.draw((0, 0))  # places the buttons

    click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1)
    assert screen.handle_event(click, screen.btn_theme.center)
    assert screen.theme_mode == "dark"
    assert screen.handle_event(click, screen.btn_acquisition.center)
    assert started.wait(2.0) and screen.acquisition_pressed
    assert screen.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1), (0, 0))
    assert not screen.acquisition_pressed
    assert not screen.handle_event(click, screen.btn_quit.center)
    pygame.quit()
