# Projet_IoT : Radar modulaire à base de SDR et d'IA

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

### Documentation SDR (pysdr)

Pour la configuration liée au matériel SDR (drivers, accès USB depuis WSL2/VM, dépendances spécifiques Pluto/RTL-SDR, etc.), suivez la documentation **pysdr** :
- `https://pysdr.org/`