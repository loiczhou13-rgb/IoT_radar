# Projet_IoT : Radar modulaire à base de SDR et d'IA

## MicroDoppler (détection respiration) — aperçu

Objectif court terme (phase 1) : **détecter une respiration** (mouvement périodique de faible amplitude) dans un contexte **séisme / avalanche**, via une signature **micro‑Doppler** (spectrogramme) et une **détection de périodicité**.  
Objectif long terme (phase 2) : **localiser** (distance/position) — non implémenté pour l’instant.

- **Portée** : en **CW micro‑Doppler** (phase 1), on **ne mesure pas de distance**. L’interface affiche plutôt des métriques utiles (résolution STFT \(\Delta t, \Delta f, \Delta v\), SNR, statut `PRESENT/ABSENT`).
- **PlutoSDR hacké** : on tient compte des limites **BW \(\le\) ~56 MHz** et **FS limitée**. Si la configuration demandée dépasse ces limites, l’app devra afficher un **warning** (valeur demandée, limite, valeur appliquée).

Documentation détaillée de l’architecture MicroDoppler : `microdoppler/README.md`.
Lancement (quand les dépendances sont installées) : `python -m microdoppler`.

## Environnement de développement (Windows/WSL2 ou VM Linux)

Cette première étape sert à obtenir un Linux “propre” (WSL2 sous Windows ou VM Linux) avec Python et les dépendances système nécessaires pour exécuter les scripts. Les dépendances SDR (PlutoSDR/RTL-SDR, drivers, etc.) dépendent fortement du matériel : pour celles-ci, référez-vous ensuite à la documentation **pysdr**.

### Option A — Windows + WSL2 (recommandé)

- **WSL2 + Ubuntu** : installez WSL2 et une distro Ubuntu (via Microsoft Store).
- **Ouvrez un terminal Ubuntu** et mettez à jour le système :

```bash
sudo apt update && sudo apt upgrade -y
```

### Option B — Linux natif ou Linux dans une VM

- **Distrib conseillée** : Ubuntu/Debian (les commandes ci-dessous sont pour `apt`).
- **Mise à jour** :

```bash
sudo apt update && sudo apt upgrade -y
```

### Paquets à installer (base)

Installez Python, l’outillage de build, et quelques libs fréquemment nécessaires (notamment pour `numpy/matplotlib` et les bindings SDR).

```bash
sudo apt install -y \
  git \
  python3 python3-pip python3-venv \
  build-essential pkg-config cmake \
  libusb-1.0-0-dev
```

### Environnement Python (venv) + dépendances

Créez un environnement virtuel puis installez les dépendances Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Si vous travaillez sur la partie Pluto (`PlutoWallDetection/PlutoFMCW`) :

```bash
pip install -r PlutoWallDetection/PlutoFMCW/requirements.txt
```

Si vous travaillez sur le module MicroDoppler (nouveau, indépendant) :

```bash
# à préciser quand le module sera ajouté au repo
python -m microdoppler
```

### Documentation SDR (pysdr)

Pour la configuration liée au matériel SDR (drivers, accès USB depuis WSL2/VM, dépendances spécifiques Pluto/RTL-SDR, etc.), suivez la documentation **pysdr** :
- `https://pysdr.org/`