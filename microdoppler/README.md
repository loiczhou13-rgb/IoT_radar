# MicroDoppler (PlutoSDR) — Architecture & modules

Ce module vise la **détection de respiration** (mouvements périodiques de faible amplitude) via **micro‑Doppler** en temps réel.

## Périmètre

- **Phase 1 (maintenant)** : détection `PRESENT/ABSENT` + `confidence` + estimation respiration (Hz/BPM) si fiable.
- **Phase 2 (plus tard)** : localisation (distance/position) — **hors scope**. Un dossier `localization/` sert de placeholder.

## Notions clés (d’après le cours/PDF)

- **Micro‑Doppler** : signature Doppler étalée autour de \(f_D\), due aux micro‑mouvements (respiration).
- **Spectrogramme (STFT)** : visualisation temps‑fréquence pour des signaux non stationnaires.
- **Suppression du clutter (avant STFT)** :
  - soustraction moyenne (clutter fixe)
  - filtre IIR 1er ordre (clutter variable) avec \(\alpha \in [0.9, 0.999]\)
  - MTI \(x[n]-x[n-1]\) (simple/efficace)

## Contraintes PlutoSDR hacké

- **BW instantanée** : \(\le\) ~56 MHz
- **FS** : limitée (choisir une valeur stable pour éviter overruns/drops)
- L’app doit **warn** si un paramètre demandé dépasse les limites (valeur demandée, limite, valeur appliquée si clamp).

## Organisation des fichiers

Le package vit à la racine du repo :

- `microdoppler/`
  - `__main__.py`
    - point d’entrée : `python -m microdoppler`
  - `main.py`
    - orchestration : init SDR, génération TX, boucle RX, pipeline DSP, UI
  - `config.py`
    - paramètres radio (FC/FS/buffer), STFT (window/overlap/nfft), détection (bandes, seuils, durées)
  - `sdr_config.py`
    - configuration Pluto (LO, FS, gains, buffers, cyclic TX)
    - validations BW/FS + warning si hors limites
  - `sdr_io.py`
    - wrapper I/O Pluto : `rx()` / `tx()` / `stop_tx()`
  - `tx_waveforms.py`
    - génération de signaux TX (multi‑types)
    - **par défaut** : `cw_tone` (phase 1)
  - `domain/targets.py`
    - presets “respiration” : plages attendues (Hz/BPM), bandes vitesse, durées mini de stabilité
  - `processing/`
    - `clutter_filter.py`
      - suppression DC + clutter (moyenne / IIR / MTI)
    - `microdoppler_stft.py`
      - STFT + conversion Doppler -> vitesse \(v=\lambda f_d/2\)
      - calcule aussi les résolutions utiles : \(\Delta t\), \(\Delta f\), \(\Delta v\)
    - `microdoppler_features.py`
      - features robustes/maintenables (énergie bande, centroid, ratio SNR, peak‑track)
    - `respiration_periodicity.py`
      - détection de périodicité (autocorr / FFT) sur la série temporelle de feature(s)
    - `respiration_detector.py`
      - décision : combine features + stabilité périodicité -> `is_present`, `confidence`, `rate_bpm`
  - `ui.py`
    - UI minimale (debug/validation) :
      - spectre TX (fréquentiel)
      - spectre RX (fréquentiel)
      - micro‑doppler (spectrogramme vitesse)
      - panneau métriques (\(\Delta t,\Delta f,\Delta v\), limites vitesse, SNR, `PRESENT/ABSENT`)
  - `localization/README.md`
    - placeholder phase 2 (API/design, pas d’implémentation)

## Interfaces et dépendances (flux de données)

```text
SDR (rx IQ) -> clutter_filter -> STFT microdoppler -> features -> periodicity -> respiration_detector
                                                             \-> UI (spectrogramme + métriques)
TX waveforms -> SDR (tx cyclic)
```

## Lancer (prévu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy matplotlib pyadi-iio

python -m microdoppler --uri ip:192.168.2.1 --fc 2.4e9 --fs 600000 --waveform cw_tone
```

