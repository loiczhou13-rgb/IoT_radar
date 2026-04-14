## Interface locale (FFT TX/RX + Range Profile)

Ce mini-projet fournit une **interface locale** (Matplotlib) pour visualiser un radar FMCW basé sur **ADALM‑Pluto (PlutoSDR)**.

Il s’inspire de la démarche “pas à pas” du chapitre PlutoSDR de PySDR (installation/connexion/test avant de lancer une appli) :
[PySDR – PlutoSDR en Python](https://pysdr.org/fr/content-fr/pluto.html)

### Fonctionnalités

- **FFT du signal émis (TX)** et **FFT du signal reçu (RX)** (domaine fréquentiel)
- **Range profile** (distance) + **détection de mouvement** via `variation` (écart-type temporel du profil MTI)
- **Bandeau d’infos** (bas de fenêtre) : `fs`, `fc`, `BW`, \(\Delta R \approx c/(2BW)\), \(R_{max}\) approx, seuil et nombre de pics détectés

### Pré-requis

- Un Pluto accessible en **USB/Ethernet gadget** (IP typique `192.168.2.1`)
- Python 3 + `pyadi-iio` (et donc `pylibiio`)

### 1) Installation Python (recommandé : venv dans le repo)

```bash
cd /home/zhoul/iot-radar
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r PlutoWallDetection/PlutoFMCW/requirements.txt
```

### 2) Vérifier la connexion Pluto (avant de lancer l’UI)

- **Ping** (si connexion réseau via USB) :

```bash
ping -c 1 192.168.2.1
```

- **Test Python minimal** (équivalent au test recommandé dans PySDR) :

```bash
./.venv/bin/python - <<'PY'
import adi
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(2.5e6)
iq = sdr.rx()
print("RX ok, nb échantillons:", len(iq))
PY
```

Si ce test échoue, corrige d’abord l’accès device (drivers/libiio/pylibiio/IP) avant de continuer.

### 3) Lancement de l’interface

```bash
./.venv/bin/python PlutoWallDetection/PlutoFMCW/fmcw_app.py
```

Si tu vois `ERREUR MATÉRIELLE : No device found`, vérifie :
- que le Pluto est joignable (par défaut `ip:192.168.2.1`)
- que `pyadi-iio` / `pylibiio` voient bien le device (réseau/USB, drivers)

### Contrôles clavier

- **e** : focus FFT émise
- **r** : focus FFT reçue
- **p** : focus range profile
- **+ / -** : zoom horizontal sur l’axe “focus”
- **i** : reset zoom
- **c** : reset MTI/background + efface les pics détectés
- **q** : quitter

### Spécifications techniques (ADALM‑Pluto / PlutoSDR “basique”)

D’après les spécifications et retours d’usage courants (et rappelées dans PySDR) :
- **Bande de fréquences RF** : environ **325 MHz → 3.8 GHz**
  - Extensions “hack” souvent citées : ~**70 MHz → 6 GHz** (non garanties, dépend firmware/config) — voir la section “Hacker PlutoSDR…” dans PySDR.
- **Chaînes** : **1 RX / 1 TX**, I/Q complexes
- **Bande passante instantanée** : typiquement **jusqu’à ~20 MHz** (souvent plus selon config/firmware)
- **Taux d’échantillonnage** : jusqu’à **~61.44 MS/s** (en pratique dépend de l’hôte/USB/CPU)
- **Convertisseurs** : 12 bits (I/Q)

### Leviers pour améliorer la résolution ou la portée (FMCW)

- **Résolution en distance** \(\Delta R \approx c/(2BW)\)
  - **Augmenter la bande balayée `BW`** améliore directement la résolution.
  - Attention : plus de `BW` peut réduire le SNR par bin et augmenter les exigences en débit/traitement.

- **Portée max affichée (dans cette implémentation)**
  - Le profil distance est limité par le nombre de bins : \(R_{max} \approx (N/2)\cdot c/(2BW)\).
  - **Augmenter `n_samples`** augmente \(R_{max}\) (et augmente la durée d’observation/chirp si `fs` fixe).

- **Détection de mouvement (robustesse)**
  - **SNR** : gains TX/RX (sans saturer), meilleures antennes, pertes RF, environnement.
  - **MTI** : le paramètre `mti_alpha` dans `fmcw_processor.py` contrôle la vitesse d’adaptation du fond statique.
  - **Fenêtrage** : Hann/Blackman aident à réduire les lobes secondaires (moins de faux pics).

- **Paramètres système**
  - **Choix de `fc`** : peut aider selon la scène (micro‑mouvements vs pertes vs réglementation).
  - **Réglementation** : l’émission RF est encadrée (puissance/bandes). À respecter.

