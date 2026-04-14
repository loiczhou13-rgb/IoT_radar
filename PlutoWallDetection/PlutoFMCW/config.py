# PlutoFMCW/config.py

class FMCWConfig:
    fs = 60_000_000           # fréquence d'échantillonnage (Hz)
    fc = 2_500_000_000       # fréquence porteuse (Hz)
    n_samples = 4096         # points par trame
    bw = 30_000_000          # bande passante chirp (Hz)
    T = n_samples / fs         # durée chirp (s)

    # gains (dB)
    tx_gain = 0
    rx_gain = 30

    # historique et affichage
    n_history = 100          # nombre de trames dans le waterfall

    # affichage spectre
    spec_ylim = (-110, -50)
    spec_xlim = (0, 15)
    waterfall_vmin = -105
    waterfall_vmax = -70

    # autres paramètres GUI
    figure_size = (12, 10)
    update_interval_ms = 30
