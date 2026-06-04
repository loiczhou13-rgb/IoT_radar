import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#from main import main

# ================= CONFIG =================
WIDTH, HEIGHT = 1000, 700
PADDING = 40
CARD_WIDTH = 420
BUTTON_WIDTH = 240
BUTTON_HEIGHT = 48

# ================= COLORS (Dark/Light Mode) =================
# Light Mode
LIGHT_BG_PRIMARY = "#f5f7fa"
LIGHT_BG_SECONDARY = "#ffffff"
LIGHT_TEXT_PRIMARY = "#212529"
LIGHT_TEXT_SECONDARY = "#6c757d"
LIGHT_CARD_BG = "#ffffff"
LIGHT_CARD_BORDER = "#00dcf0"
LIGHT_BUTTON_IDLE = "#e9ecef"
LIGHT_BUTTON_HOVER = "#dee2e6"
LIGHT_BUTTON_BORDER = "#00dcf0"
LIGHT_FLUO_CYAN = "#00dcf0"
LIGHT_PLOT_BG = "#ffffff"  # Fond blanc pour le graphique en light mode

# Dark Mode
DARK_BG_PRIMARY = "#08080c"
DARK_BG_SECONDARY = "#121219"
DARK_TEXT_PRIMARY = "#f0f0ff"
DARK_TEXT_SECONDARY = "#9696b4"
DARK_CARD_BG = "#121219"
DARK_CARD_BORDER = "#0078c8"
DARK_BUTTON_IDLE = "#191922"
DARK_BUTTON_HOVER = "#282835"
DARK_BUTTON_BORDER = "#0078c8"
DARK_FLUO_CYAN = "#00dcf0"
DARK_PLOT_BG = "#121219"  # Fond sombre pour le graphique en dark mode

# ================= STATE =================
mode = "distance"
screen_state = "main"
is_dark_mode = False

# ================= TKINTER SETUP =================
root = tk.Tk()
root.title("Cyber Dashboard")
root.geometry(f"{WIDTH}x{HEIGHT}")
root.resizable(False, False)

# ================= THEME MANAGER =================
def get_color(key):
    if is_dark_mode:
        colors = {
            "bg_primary": DARK_BG_PRIMARY,
            "bg_secondary": DARK_BG_SECONDARY,
            "text_primary": DARK_TEXT_PRIMARY,
            "text_secondary": DARK_TEXT_SECONDARY,
            "card_bg": DARK_CARD_BG,
            "card_border": DARK_CARD_BORDER,
            "button_idle": DARK_BUTTON_IDLE,
            "button_hover": DARK_BUTTON_HOVER,
            "button_border": DARK_BUTTON_BORDER,
            "fluo_cyan": DARK_FLUO_CYAN,
            "plot_bg": DARK_PLOT_BG,
        }
    else:
        colors = {
            "bg_primary": LIGHT_BG_PRIMARY,
            "bg_secondary": LIGHT_BG_SECONDARY,
            "text_primary": LIGHT_TEXT_PRIMARY,
            "text_secondary": LIGHT_TEXT_SECONDARY,
            "card_bg": LIGHT_CARD_BG,
            "card_border": LIGHT_CARD_BORDER,
            "button_idle": LIGHT_BUTTON_IDLE,
            "button_hover": LIGHT_BUTTON_HOVER,
            "button_border": LIGHT_BUTTON_BORDER,
            "fluo_cyan": LIGHT_FLUO_CYAN,
            "plot_bg": LIGHT_PLOT_BG,
        }
    return colors.get(key, "#000000")

def apply_theme():
    root.configure(bg=get_color("bg_primary"))
    main_frame.configure(bg=get_color("bg_primary"))
    menu_frame.configure(bg=get_color("bg_primary"))
    card.configure(bg=get_color("card_bg"))
    card_menu.configure(bg=get_color("card_bg"))
    title_label.configure(fg=get_color("text_primary"), bg=get_color("card_bg"))
    title_label_menu.configure(fg=get_color("text_primary"), bg=get_color("card_bg"))
    mode_badge.configure(fg=get_color("fluo_cyan"), bg=get_color("card_bg"))
    footer_label.configure(fg=get_color("text_secondary"), bg=get_color("card_bg"))
    for btn in [btn_plot, btn_mode, btn_distance, btn_precision, btn_back, theme_btn]:
        btn.configure(
            bg=get_color("button_idle"),
            fg=get_color("text_primary"),
            activebackground=get_color("button_hover"),
            activeforeground=get_color("text_primary"),
            highlightbackground=get_color("button_border")
        )
    card.configure(highlightbackground=get_color("card_border"))
    card_menu.configure(highlightbackground=get_color("card_border"))

def toggle_theme():
    global is_dark_mode
    is_dark_mode = not is_dark_mode
    apply_theme()
    theme_btn.configure(text="Mode clair" if is_dark_mode else "Mode sombre")

# ================= MATPLOTLIB IN NEW TK WINDOW =================
def open_matplotlib():
    plot_window = tk.Toplevel(root)
    plot_window.title("Real-Time Sine Wave")
    plot_window.geometry("800x600")
    plot_window.configure(bg=get_color("bg_primary"))

    fig = plt.Figure(figsize=(8, 6), facecolor=get_color("plot_bg"))
    ax = fig.add_subplot(111, facecolor=get_color("plot_bg"))
    ax.tick_params(colors=get_color("text_secondary"))
    ax.grid(color=get_color("card_border"), alpha=0.5)

    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, plot_window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    x, y = [], []
    def animate():
        nonlocal x, y
        x.append(len(x))
        y.append(np.sin(len(x) * 0.1))
        ax.clear()
        ax.plot(x, y, color=get_color("fluo_cyan"), linewidth=2.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("Real-Time Sine Wave", color=get_color("fluo_cyan"), pad=20, fontweight='light')
        ax.tick_params(colors=get_color("text_secondary"))
        ax.grid(color=get_color("card_border"), alpha=0.5)
        ax.set_facecolor(get_color("plot_bg"))
        fig.set_facecolor(get_color("plot_bg"))
        canvas.draw()
        if len(x) < 200:
            plot_window.after(30, animate)

    animate()

# ================= SCREEN MANAGEMENT =================
def show_main():
    global screen_state
    screen_state = "main"
    main_frame.pack(fill=tk.BOTH, expand=True)
    menu_frame.pack_forget()

def show_menu():
    global screen_state
    screen_state = "menu"
    main_frame.pack_forget()
    menu_frame.pack(fill=tk.BOTH, expand=True)

def set_mode(new_mode):
    global mode
    mode = new_mode
    mode_badge.configure(text=mode.upper())
    show_main()

# ================= UI SETUP =================
# Main Frame
main_frame = tk.Frame(root, bg=get_color("bg_primary"))
main_frame.pack(fill=tk.BOTH, expand=True)

# Menu Frame
menu_frame = tk.Frame(root, bg=get_color("bg_primary"))

# Card (Main)
card = tk.Frame(
    main_frame,
    bg=get_color("card_bg"),
    highlightthickness=1,
    highlightbackground=get_color("card_border"),
    padx=PADDING,
    pady=PADDING,
    width=CARD_WIDTH,
    height=520
)
card.pack(pady=50)
card.pack_propagate(False)

# Title (Main)
title_label = tk.Label(
    card,
    text="Radar Modulaire",
    font=("Segoe UI Light", 24),
    fg=get_color("text_primary"),
    bg=get_color("card_bg")
)
title_label.pack(pady=(20, 10))

# Mode Badge (Main)
mode_badge = tk.Label(
    card,
    text=mode.upper(),
    font=("Segoe UI Semibold", 12),
    fg=get_color("fluo_cyan"),
    bg=get_color("card_bg"),
    relief=tk.FLAT,
    padx=20,
    pady=5
)
mode_badge.pack(pady=10)

# Buttons (Main)
btn_plot = tk.Button(
    card,
    text="Lancer l'acquisition",
    font=("Segoe UI Semibold", 12),
    bg=get_color("button_idle"),
    fg=get_color("text_primary"),
    activebackground=get_color("button_hover"),
    activeforeground=get_color("text_primary"),
    relief=tk.FLAT,
    highlightthickness=1,
    highlightbackground=get_color("button_border"),
    width=20,
    height=2,
    command=open_matplotlib
)
btn_plot.pack(pady=10)

btn_mode = tk.Button(
    card,
    text="Changer de Mode",
    font=("Segoe UI Semibold", 12),
    bg=get_color("button_idle"),
    fg=get_color("text_primary"),
    activebackground=get_color("button_hover"),
    activeforeground=get_color("text_primary"),
    relief=tk.FLAT,
    highlightthickness=1,
    highlightbackground=get_color("button_border"),
    width=20,
    height=2,
    command=show_menu
)
btn_mode.pack(pady=10)

# Footer (Main)
footer_label = tk.Label(
    card,
    text="Projet S6 - Pôle IOT - 2026",
    font=("Segoe UI", 10),
    fg=get_color("text_secondary"),
    bg=get_color("card_bg")
)
footer_label.pack(pady=(20, 0))

# Card (Menu)
card_menu = tk.Frame(
    menu_frame,
    bg=get_color("card_bg"),
    highlightthickness=1,
    highlightbackground=get_color("card_border"),
    padx=PADDING,
    pady=PADDING,
    width=CARD_WIDTH,
    height=440
)
card_menu.pack(pady=50)
card_menu.pack_propagate(False)

# Title (Menu)
title_label_menu = tk.Label(
    card_menu,
    text="Sélection Mode",
    font=("Segoe UI Light", 24),
    fg=get_color("text_primary"),
    bg=get_color("card_bg")
)
title_label_menu.pack(pady=(20, 20))

# Buttons (Menu)
btn_distance = tk.Button(
    card_menu,
    text="Mode Distance",
    font=("Segoe UI Semibold", 12),
    bg=get_color("button_idle"),
    fg=get_color("text_primary"),
    activebackground=get_color("button_hover"),
    activeforeground=get_color("text_primary"),
    relief=tk.FLAT,
    highlightthickness=1,
    highlightbackground=get_color("button_border"),
    width=20,
    height=2,
    command=lambda: set_mode("distance")
)
btn_distance.pack(pady=10)

btn_precision = tk.Button(
    card_menu,
    text="Mode Précision",
    font=("Segoe UI Semibold", 12),
    bg=get_color("button_idle"),
    fg=get_color("text_primary"),
    activebackground=get_color("button_hover"),
    activeforeground=get_color("text_primary"),
    relief=tk.FLAT,
    highlightthickness=1,
    highlightbackground=get_color("button_border"),
    width=20,
    height=2,
    command=lambda: set_mode("precision")
)
btn_precision.pack(pady=10)

btn_back = tk.Button(
    card_menu,
    text="Retour",
    font=("Segoe UI Semibold", 12),
    bg=get_color("button_idle"),
    fg=get_color("text_primary"),
    activebackground=get_color("button_hover"),
    activeforeground=get_color("text_primary"),
    relief=tk.FLAT,
    highlightthickness=1,
    highlightbackground=get_color("button_border"),
    width=20,
    height=2,
    command=show_main
)
btn_back.pack(pady=10)

# Theme Toggle Button
theme_btn = tk.Button(
    root,
    text="Mode sombre",
    font=("Segoe UI Semibold", 10),
    bg=get_color("button_idle"),
    fg=get_color("text_primary"),
    activebackground=get_color("button_hover"),
    activeforeground=get_color("text_primary"),
    relief=tk.FLAT,
    highlightthickness=1,
    highlightbackground=get_color("button_border"),
    command=toggle_theme
)
theme_btn.place(x=WIDTH-120, y=20, width=100, height=30)

# ================= INIT =================
apply_theme()
show_main()

# ================= RUN =================
root.mainloop()