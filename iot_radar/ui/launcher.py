"""Pygame home screen of the radar ("Modular Radar").

Place in the project: entry point for the end users.  The screen shows one
button per launch action (e.g. radar on the PlutoSDR, radar in simulation),
then "Dark / light theme" and "Quit".  The launcher does not know the radar
code: the actions are given by the caller (``scripts/launcher.py``, which runs
the radar in a separate process).
"""

from __future__ import annotations

import sys
from typing import Callable, Sequence

import pygame
import pygame.freetype

WIDTH, HEIGHT = 1200, 760
FPS = 60

THEMES = {
    "dark": {
        "bg": (8, 10, 18),
        "card": (18, 22, 34),
        "card_2": (26, 30, 46),
        "text": (240, 245, 255),
        "text_secondary": (145, 155, 180),
        "accent": (0, 220, 255),
        "accent_2": (170, 120, 255),
        "button": (30, 35, 50),
        "button_hover": (45, 55, 78),
        "border": (60, 90, 140),
        "shadow": (0, 0, 0, 140),
        "badge": (0, 220, 255)
    },
    "light": {
        "bg": (240, 244, 252),
        "card": (255, 255, 255),
        "card_2": (245, 248, 255),
        "text": (20, 24, 35),
        "text_secondary": (90, 100, 125),
        "accent": (0, 140, 255),
        "accent_2": (120, 90, 255),
        "button": (228, 234, 246),
        "button_hover": (210, 220, 240),
        "border": (180, 190, 220),
        "shadow": (0, 0, 0, 30),
        "badge": (0, 140, 255)
    }
}

BTN_W = 260
BTN_H = 54
BTN_SPACE = 18


def _load_fonts() -> dict[str, pygame.freetype.Font]:
    """Fonts of the screen (Segoe UI, or Arial if it fails)."""
    try:
        return {
            "title": pygame.freetype.SysFont("Segoe UI Semibold", 44),
            "subtitle": pygame.freetype.SysFont("Segoe UI", 20),
            "button": pygame.freetype.SysFont("Segoe UI Semibold", 18),
            "badge": pygame.freetype.SysFont("Segoe UI Bold", 15),
            "small": pygame.freetype.SysFont("Segoe UI", 14),
        }
    except Exception:
        return {
            "title": pygame.freetype.SysFont("Arial", 44),
            "subtitle": pygame.freetype.SysFont("Arial", 20),
            "button": pygame.freetype.SysFont("Arial", 18),
            "badge": pygame.freetype.SysFont("Arial", 15),
            "small": pygame.freetype.SysFont("Arial", 14),
        }


class HomeScreen:
    """Home screen window and its event loop.

    Parameters
    ----------
    actions : sequence of (label, callable)
        One button per action, in this order.  Each function must return
        quickly (e.g. start a separate process): the event loop waits for it.
    """

    def __init__(self, actions: Sequence[tuple[str, Callable[[], object]]]) -> None:
        pygame.init()
        pygame.freetype.init()
        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT),
            pygame.SCALED | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Modular Radar")
        self.clock = pygame.time.Clock()
        self.fonts = _load_fonts()
        self.actions = list(actions)
        self.theme_mode = "light"
        self.pressed_action: int | None = None  # highlighted while the mouse button is held
        self.action_buttons = [pygame.Rect(0, 0, BTN_W, BTN_H) for _ in self.actions]
        self.btn_quit = pygame.Rect(0, 0, BTN_W, BTN_H)
        self.btn_theme = pygame.Rect(0, 0, BTN_W, BTN_H)

    @property
    def theme(self) -> dict:
        """Colours of the active theme."""
        return THEMES[self.theme_mode]

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def draw_rounded_rect(self, rect: pygame.Rect, color, radius: int = 20) -> None:
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)

    def draw_shadow(self, rect: pygame.Rect, shadow_color) -> None:
        shadow_rect = rect.copy()
        shadow_rect.x += 6
        shadow_rect.y += 8
        shadow_surface = pygame.Surface(
            (shadow_rect.width, shadow_rect.height),
            pygame.SRCALPHA
        )
        pygame.draw.rect(
            shadow_surface,
            shadow_color,
            (0, 0, shadow_rect.width, shadow_rect.height),
            border_radius=24
        )
        self.screen.blit(shadow_surface, shadow_rect)

    def draw_card(self, rect: pygame.Rect) -> None:
        theme = self.theme
        self.draw_shadow(rect, theme["shadow"])
        self.draw_rounded_rect(rect, theme["card"], 24)
        pygame.draw.rect(self.screen, theme["border"], rect, width=1, border_radius=24)

    def draw_button(self, rect: pygame.Rect, text: str, hovered: bool = False,
                    accent: bool = False, pressed: bool = False) -> None:
        theme = self.theme
        bg = theme["button_hover"] if hovered else theme["button"]
        if accent:
            bg = theme["accent"]
        if pressed:  # highlighted while the button is held down
            bg = theme["accent_2"]
        self.draw_rounded_rect(rect, bg, 16)
        pygame.draw.rect(self.screen, theme["border"], rect, width=1, border_radius=16)
        text_color = (255, 255, 255) if accent or pressed else theme["text"]
        surf, txt_rect = self.fonts["button"].render(text, text_color)
        txt_rect.center = rect.center
        self.screen.blit(surf, txt_rect)

    def draw_badge(self, center_x: int, y: int, text: str) -> None:
        theme = self.theme
        badge_rect = pygame.Rect(0, 0, 190, 38)
        badge_rect.center = (center_x, y)
        glow = pygame.Surface((210, 58), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*theme["badge"], 40), (0, 0, 210, 58), border_radius=22)
        self.screen.blit(glow, (badge_rect.x - 10, badge_rect.y - 10))
        self.draw_rounded_rect(badge_rect, theme["card_2"], 18)
        pygame.draw.rect(self.screen, theme["accent"], badge_rect, width=1, border_radius=18)
        surf, txt_rect = self.fonts["badge"].render(text, theme["accent"])
        txt_rect.center = badge_rect.center
        self.screen.blit(surf, txt_rect)

    # ------------------------------------------------------------------
    # Events and frames
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event, mouse_pos: tuple[int, int]) -> bool:
        """Process one event; return ``False`` to quit."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            for index, (button, (_, action)) in enumerate(zip(self.action_buttons, self.actions)):
                if button.collidepoint(mouse_pos):
                    self.pressed_action = index
                    action()
            if self.btn_quit.collidepoint(mouse_pos):
                return False
            if self.btn_theme.collidepoint(mouse_pos):
                self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
        if event.type == pygame.MOUSEBUTTONUP:
            self.pressed_action = None
        return True

    def draw(self, mouse_pos: tuple[int, int]) -> None:
        """Draw one frame of the home screen."""
        theme = self.theme
        self.screen.fill(theme["bg"])

        card_rect = pygame.Rect(0, 0, 520, 560 + (BTN_H + BTN_SPACE) * max(0, len(self.actions) - 1))
        card_rect.center = (WIDTH // 2, HEIGHT // 2)
        self.draw_card(card_rect)

        title_surf, title_rect = self.fonts["title"].render("Modular Radar", theme["text"])
        title_rect.center = (WIDTH // 2, card_rect.y + 70)
        self.screen.blit(title_surf, title_rect)

        subtitle_surf, subtitle_rect = self.fonts["subtitle"].render("", theme["text_secondary"])
        subtitle_rect.center = (WIDTH // 2, title_rect.bottom + 30)
        self.screen.blit(subtitle_surf, subtitle_rect)

        self.draw_badge(WIDTH // 2, subtitle_rect.bottom + 45, "S6 project - IoT team")

        y = card_rect.y + 260
        center_x = WIDTH // 2
        for index, (button, (label, _)) in enumerate(zip(self.action_buttons, self.actions)):
            button.center = (center_x, y)
            self.draw_button(button, label, button.collidepoint(mouse_pos), accent=True,
                             pressed=self.pressed_action == index)
            y += BTN_H + BTN_SPACE
        self.btn_theme.center = (center_x, y)
        self.btn_quit.center = (center_x, y + BTN_H + BTN_SPACE)
        self.draw_button(self.btn_theme, "Dark / light theme", self.btn_theme.collidepoint(mouse_pos))
        self.draw_button(self.btn_quit, "Quit", self.btn_quit.collidepoint(mouse_pos))

        footer_surf, footer_rect = self.fonts["small"].render("2026", theme["text_secondary"])
        footer_rect.center = (WIDTH // 2, card_rect.bottom - 40)
        self.screen.blit(footer_surf, footer_rect)

        pygame.display.flip()

    def run(self) -> None:
        """Event loop, until the window is closed or "Quit" is clicked."""
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if not self.handle_event(event, mouse_pos):
                    running = False
            self.draw(mouse_pos)
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()
