import pygame
import pygame.freetype  # Pour des polices lisses
import numpy as np
import sys
import matplotlib.pyplot as plt

# Initialisation
pygame.init()
pygame.freetype.init()

# Constantes
WIDTH, HEIGHT = 1200, 800
GRAPH_WIDTH, GRAPH_HEIGHT = 350, 250
MARGIN = 60
PADDING = 50
FPS = 60
COLORS = [
    (100, 180, 220, 200),  # Bleu clair (avec transparence pour les lignes)
    (120, 200, 120, 200),  # Vert doux
    (240, 160, 120, 200),  # Orange pâle
    (180, 140, 220, 200)   # Violet clair
]

# Couleurs (thème sombre ultra-pro)
BG_COLOR = (15, 15, 20)
MENU_COLOR = (25, 25, 30)
BUTTON_COLOR = (40, 40, 45)
BUTTON_HOVER_COLOR = (70, 70, 80)
TEXT_COLOR = (210, 210, 220)
AXIS_COLOR = (80, 80, 90)
GRID_COLOR = (40, 40, 50)
HIGHLIGHT_COLOR = (255, 255, 255)
GRAPH_BG_COLOR = (30, 30, 35)

# Fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
pygame.display.set_caption("Radar Pro - Navigation par graphe")
clock = pygame.time.Clock()

# Polices lisses
try:
    title_font = pygame.freetype.Font("Arial", 36)
    button_font = pygame.freetype.Font("Arial", 20)
    label_font = pygame.freetype.Font("Arial", 18)
    graph_title_font = pygame.freetype.Font("Arial", 20)
except:
    # Fallback si la police n'est pas disponible
    title_font = pygame.freetype.Font(None, 36)
    button_font = pygame.freetype.Font(None, 20)
    label_font = pygame.freetype.Font(None, 18)
    graph_title_font = pygame.freetype.Font(None, 20)

# Variables
mode = "distance"
graphs_data = [np.random.rand(50) for _ in range(4)]
time = 0
show_menu = False
fullscreen_graph = None  # Index du graphe en plein écran (None = vue d'ensemble)

def generate_data(mode, t, graph_idx):
    """Génère des données fictives pour chaque graphe."""
    if mode == "distance":
        return [0.8 * np.sin(t * 0.5 + i * 0.1 + graph_idx) + 0.2 * np.random.normal(0, 0.05) for i in range(50)]
    else:
        return [0.5 * np.sin(t * 1.2 + i * 0.1 + graph_idx) + 0.1 * np.random.normal(0, 0.02) for i in range(50)]

def draw_graph(surface, x, y, data, color, title, is_fullscreen=False):
    """Dessine un graphe avec un design ultra-pro et aéré."""
    # Fond du graphe (ombre légère + bordure)
    if not is_fullscreen:
        pygame.draw.rect(surface, (45, 45, 50), (x-8, y-8, GRAPH_WIDTH+16, GRAPH_HEIGHT+16), border_radius=8)
    pygame.draw.rect(surface, GRAPH_BG_COLOR, (x, y, GRAPH_WIDTH, GRAPH_HEIGHT), border_radius=5)

    # Titre (police lisse)
    graph_title_font.render_to(surface, (x + 20, y - 30), title, TEXT_COLOR)

    # Axes (lignes très fines)
    pygame.draw.line(surface, AXIS_COLOR, (x, y + GRAPH_HEIGHT), (x + GRAPH_WIDTH, y + GRAPH_HEIGHT), 1)
    pygame.draw.line(surface, AXIS_COLOR, (x, y), (x, y + GRAPH_HEIGHT), 1)

    # Grille (discrète, seulement 2 lignes horizontales/verticales)
    for i in [1, 3]:
        pygame.draw.line(surface, GRID_COLOR, (x, y + i * GRAPH_HEIGHT // 4), (x + GRAPH_WIDTH, y + i * GRAPH_HEIGHT // 4), 1)
        pygame.draw.line(surface, GRID_COLOR, (x + i * GRAPH_WIDTH // 4, y), (x + i * GRAPH_WIDTH // 4, y + GRAPH_HEIGHT), 1)

    # Courbe (ligne lisse et anti-aliased)
    points = []
    for i, value in enumerate(data):
        px = x + i * GRAPH_WIDTH // 50
        py = y + GRAPH_HEIGHT - int((value + 1) * GRAPH_HEIGHT // 3)
        points.append((px, py))
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 2)

    # Bouton "Plein écran" (uniquement en vue d'ensemble)
    if not is_fullscreen:
        margin = 10  # marge intérieure
        button_width, button_height = 110, 30

        button_rect = pygame.Rect(
            x + GRAPH_WIDTH - button_width - margin,
            y + GRAPH_HEIGHT - button_height - margin,
            button_width,
            button_height
        )

        pygame.draw.rect(surface, BUTTON_COLOR, button_rect, border_radius=5)
        pygame.draw.rect(surface, HIGHLIGHT_COLOR, button_rect, 1, border_radius=5)

        text_surf, text_rect = label_font.render("Plein écran", TEXT_COLOR)
        text_rect.center = button_rect.center
        surface.blit(text_surf, text_rect)

        return button_rect
def open_matplotlib_live():
    plt.figure("Courbe en temps réel")
    x, y = [], []

    for i in range(200):
        x.append(i)
        y.append(np.sin(i * 0.1))

        plt.clf()
        plt.plot(x, y)
        plt.ylim(-1.5, 1.5)
        plt.title("Courbe en temps réel")
        plt.pause(0.05)

        # Vérifie si la fenêtre est fermée
        if not plt.fignum_exists("Courbe en temps réel"):
            break

    plt.close()

def draw_button(text, x, y, w, h, is_hover=False):
    """Dessine un bouton ultra-pro avec effet hover."""
    color = BUTTON_HOVER_COLOR if is_hover else BUTTON_COLOR
    pygame.draw.rect(screen, (80, 80, 90), (x-2, y-2, w+4, h+4), border_radius=6)  # Ombre
    pygame.draw.rect(screen, color, (x, y, w, h), border_radius=5)
    pygame.draw.rect(screen, HIGHLIGHT_COLOR, (x, y, w, h), 1, border_radius=5)

    text_surf, _ = button_font.render(text, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=(x + w//2, y + h//2))
    screen.blit(text_surf, text_rect)

    return pygame.Rect(x, y, w, h)

def draw_main_screen():
    """Dessine l'écran principal avec les 4 graphes et boutons plein écran."""
    screen.fill(BG_COLOR)

    # Titre principal (police lisse)
    title_font.render_to(screen, (MARGIN, MARGIN//2), "Radar Pro - Vue d'ensemble", HIGHLIGHT_COLOR)

    # Sous-titre (mode actuel)
    label_font.render_to(screen, (MARGIN, MARGIN//2 + 40), f"Mode: {mode.upper()}", TEXT_COLOR)

    # Graphes + boutons plein écran
    buttons = []
    for i, (data, color) in enumerate(zip(graphs_data, COLORS)):
        x = MARGIN + i % 2 * (GRAPH_WIDTH + PADDING)
        y = 120 + i // 2 * (GRAPH_HEIGHT + PADDING)
        button = draw_graph(screen, x, y, data, color, f"Cible {i+1}")
        buttons.append((button, i))

    # Bouton pour changer de mode
    mode_button = draw_button("Changer de mode", WIDTH - 220, HEIGHT - 80, 200, 50)
    matplotlib_button = draw_button("Matplotlib Live", WIDTH - 220, HEIGHT - 150, 200, 50)

    return buttons, mode_button, matplotlib_button

def draw_fullscreen_graph(graph_idx):
    """Dessine un graphe en plein écran (simulation pysdr)."""
    screen.fill(BG_COLOR)
    data = graphs_data[graph_idx]
    color = COLORS[graph_idx]

    # Titre
    title_font.render_to(screen, (MARGIN, MARGIN//2), f"Radar Pro - Cible {graph_idx+1} (Mode: {mode})", HIGHLIGHT_COLOR)

    # Graphe en grand
    draw_graph(screen, WIDTH//2 - GRAPH_WIDTH//2 - 20, HEIGHT//2 - GRAPH_HEIGHT//2, data, color, f"Cible {graph_idx+1} (Plein écran)", True)

    # Bouton retour
    back_button = draw_button("Retour à la vue d'ensemble", WIDTH//2 - 150, HEIGHT - 80, 300, 50)

    return back_button

def draw_menu_screen():
    """Dessine l'écran de sélection de mode (design pro)."""
    screen.fill(MENU_COLOR)

    # Titre
    title_font.render_to(screen, (WIDTH//2 - 100, 100), "Sélection du mode", HIGHLIGHT_COLOR)

    # Boutons de mode
    button_distance = draw_button("Mode Distance", WIDTH//2 - 100, 250, 200, 60)
    button_precision = draw_button("Mode Précision", WIDTH//2 - 100, 350, 200, 60)
    button_back = draw_button("Retour", WIDTH//2 - 100, 450, 200, 60)

    return button_distance, button_precision, button_back

# Boucle principale
running = True
open_plot = False
while running:
    mouse_pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if fullscreen_graph is not None:
                # En mode plein écran, seul le bouton "Retour" est actif
                back_button = draw_fullscreen_graph(fullscreen_graph)
                if back_button.collidepoint(mouse_pos):
                    fullscreen_graph = None
            elif show_menu:
                # En mode menu
                button_distance, button_precision, button_back = draw_menu_screen()
                if button_distance.collidepoint(mouse_pos):
                    mode = "distance"
                    show_menu = False
                if button_precision.collidepoint(mouse_pos):
                    mode = "precision"
                    show_menu = False
                if button_back.collidepoint(mouse_pos):
                    show_menu = False
            else:
                # En mode vue d'ensemble
                buttons, mode_button, matplotlib_button = draw_main_screen()
                if matplotlib_button.collidepoint(mouse_pos):
                    open_plot = True
                if mode_button.collidepoint(mouse_pos):
                    show_menu = True
                else:
                    for button, graph_idx in buttons:
                        if button and button.collidepoint(mouse_pos):
                            fullscreen_graph = graph_idx

    # Mise à jour des données
    time += 0.05
    for i in range(len(graphs_data)):
        graphs_data[i] = generate_data(mode, time, i)

    # Dessin
    if fullscreen_graph is not None:
        back_button = draw_fullscreen_graph(fullscreen_graph)
        # Effet hover sur le bouton retour
        is_hover = back_button.collidepoint(mouse_pos)
        draw_button("Retour à la vue d'ensemble", back_button.x, back_button.y, back_button.w, back_button.h, is_hover)
    elif show_menu:
        button_distance, button_precision, button_back = draw_menu_screen()
        # Effet hover sur les boutons du menu
        is_hover = button_distance.collidepoint(mouse_pos)
        draw_button("Mode Distance", button_distance.x, button_distance.y, button_distance.w, button_distance.h, is_hover)
        is_hover = button_precision.collidepoint(mouse_pos)
        draw_button("Mode Précision", button_precision.x, button_precision.y, button_precision.w, button_precision.h, is_hover)
        is_hover = button_back.collidepoint(mouse_pos)
        draw_button("Retour", button_back.x, button_back.y, button_back.w, button_back.h, is_hover)
    else:
        buttons, mode_button, matplotlib_button = draw_main_screen()
        # Effet hover sur le bouton mode
        is_hover = mode_button.collidepoint(mouse_pos)
        draw_button("Changer de mode", mode_button.x, mode_button.y, mode_button.w, mode_button.h, is_hover)
        is_hover = matplotlib_button.collidepoint(mouse_pos)
        draw_button("Matplotlib Live", matplotlib_button.x, matplotlib_button.y, matplotlib_button.w, matplotlib_button.h, is_hover)
        # Effet hover sur les boutons plein écran
        for button, _ in buttons:
            if button and button.collidepoint(mouse_pos):
                pygame.draw.rect(screen, BUTTON_HOVER_COLOR, button, border_radius=3)

                text_surf, text_rect = label_font.render("Plein écran", TEXT_COLOR)
                text_rect.center = button.center  

                screen.blit(text_surf, text_rect)
    pygame.display.flip()
    clock.tick(FPS)
    if open_plot:
        open_matplotlib_live()
        open_plot = False

pygame.quit()
sys.exit()