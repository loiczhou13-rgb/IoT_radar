"""Pygame home screen of the radar ("Radar Modulaire").

The screen shows three buttons: start the acquisition, toggle the light / dark
theme, quit.  The launcher does not know the radar code: the function that
starts the acquisition is given by the caller (``scripts/launcher.py``).
"""

from __future__ import annotations

import sys
import threading
from typing import Callable

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
    start_acquisition : callable
        Function run (in a background thread) by the "Lancer Acquisition"
        button.
    """

    def __init__(self, start_acquisition: Callable[[], None]) -> None:
        pygame.init()
        pygame.freetype.init()
        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT),
            pygame.SCALED | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Radar Modulaire")
        self.clock = pygame.time.Clock()
        self.fonts = _load_fonts()
        self.start_acquisition = start_acquisition
        self.theme_mode = "light"
        self.acquisition_pressed = False  # "pressed" look of the Acquisition button
        self.btn_acquisition = pygame.Rect(0, 0, BTN_W, BTN_H)
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
        if pressed:  # "open" look while the button is pressed
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
            if self.btn_acquisition.collidepoint(mouse_pos):
                self.acquisition_pressed = True
                threading.Thread(target=self.start_acquisition, daemon=True).start()
            elif self.btn_quit.collidepoint(mouse_pos):
                return False
            elif self.btn_theme.collidepoint(mouse_pos):
                self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
        if event.type == pygame.MOUSEBUTTONUP and self.acquisition_pressed:
            self.acquisition_pressed = False
        return True

    def draw(self, mouse_pos: tuple[int, int]) -> None:
        """Draw one frame of the home screen."""
        theme = self.theme
        self.screen.fill(theme["bg"])

        card_rect = pygame.Rect(0, 0, 520, 560)
        card_rect.center = (WIDTH // 2, HEIGHT // 2)
        self.draw_card(card_rect)

        title_surf, title_rect = self.fonts["title"].render("Radar Modulaire", theme["text"])
        title_rect.center = (WIDTH // 2, card_rect.y + 70)
        self.screen.blit(title_surf, title_rect)

        subtitle_surf, subtitle_rect = self.fonts["subtitle"].render("", theme["text_secondary"])
        subtitle_rect.center = (WIDTH // 2, title_rect.bottom + 30)
        self.screen.blit(subtitle_surf, subtitle_rect)

        self.draw_badge(WIDTH // 2, subtitle_rect.bottom + 45, "Projet S6 - Pôle IoT")

        start_y = card_rect.y + 260
        center_x = WIDTH // 2
        self.btn_acquisition.center = (center_x, start_y)
        self.btn_theme.center = (center_x, start_y + BTN_H + BTN_SPACE)
        self.btn_quit.center = (center_x, start_y + (BTN_H + BTN_SPACE) * 2)
        self.draw_button(self.btn_acquisition, "Lancer Acquisition",
                         self.btn_acquisition.collidepoint(mouse_pos), accent=True,
                         pressed=self.acquisition_pressed)
        self.draw_button(self.btn_theme, "Mode sombre / clair", self.btn_theme.collidepoint(mouse_pos))
        self.draw_button(self.btn_quit, "Quitter", self.btn_quit.collidepoint(mouse_pos))

        footer_surf, footer_rect = self.fonts["small"].render("2026", theme["text_secondary"])
        footer_rect.center = (WIDTH // 2, card_rect.bottom - 40)
        self.screen.blit(footer_surf, footer_rect)

        pygame.display.flip()

    def run(self) -> None:
        """Event loop, until the window is closed or "Quitter" is clicked."""
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
