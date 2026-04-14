# Radar Micro-Doppler — Détection de survivants ensevelis

Système radar portable à onde continue (CW) pour détecter la respiration
de personnes ensevelies sous des décombres (séisme, avalanche, effondrement).

Matériel : **PlutoSDR** (ADALM-PLUTO, AD9363) + **Raspberry Pi**.

---

## Contexte physique

Un signal CW à **f_c = 2.4 GHz** (λ ≈ 12.5 cm) est émis vers la zone de
recherche. Le thorax d'un survivant oscille à **f_v ≈ 0.2–0.5 Hz** avec une
amplitude **D ≈ 5–15 mm**. Ce mouvement module involontairement la phase du
signal réfléchi (modulation PM). En bande de base, le signal reçu s'écrit :

```
x(t) = exp(j·φ(t))     avec  φ(t) = φ₀ − (4πD/λ)·sin(2π·f_v·t)
```

Le développement en **séries de Bessel** révèle un peigne de raies spectrales
à ±n·f_v. La raie fondamentale (±f_v) porte l'essentiel de l'énergie
respiratoire (J₁(m) ≈ 0.44 pour un indice de modulation m ≈ 1).

---

## Architecture du pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  emission.py │────▶│ acquisition.py│────▶│ decimation.py│
│  Signal TX   │     │  IQ (PlutoSDR │     │ ↓ f_s par D  │
│  CW / offset │     │  ou simulation)    │              │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌──────────────┐     ┌───────▼──────┐
                    │ windowing.py │◀────│  clutter.py  │
                    │ Fenêtre STFT │     │ Suppression  │
                    └──────┬───────┘     │ composante DC│
                           │             └──────────────┘
                    ┌──────▼───────┐
                    │spectrogramme │
                    │    .py       │
                    │ STFT → Z, dB │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │     detection.py        │
              │ SNR bande/référence     │
              │ → alerte respiration    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    utils/display.py     │
              │  Dashboard matplotlib   │
              │  6 panneaux temps réel  │
              └─────────────────────────┘
```

Orchestration dans **`main.py`** :
emission → acquisition → décimation → clutter → spectrogramme (+ fenêtrage) → détection → affichage.

---

## Installation

### Prérequis système

```bash
# libiio — bibliothèque de communication avec le PlutoSDR
sudo apt install libiio-dev libiio-utils

# Vérifier que le PlutoSDR est détecté
iio_info -s
```

### Dépendances Python

```bash
pip install numpy scipy matplotlib pyyaml pyadi-iio
```

| Paquet        | Rôle                                               |
|---------------|-----------------------------------------------------|
| `numpy`       | Calcul vectoriel, tableaux IQ                       |
| `scipy`       | Décimation, fenêtrage, STFT                         |
| `matplotlib`  | Dashboard temps réel (FuncAnimation)                |
| `pyyaml`      | Lecture de `config.yaml`                             |
| `pyadi-iio`   | Interface Python pour le PlutoSDR via libiio         |

### Mise à jour firmware PlutoSDR (optionnel)

Si le firmware du Pluto est ancien, le taux d'échantillonnage minimal peut
être supérieur à 521 kHz. Consulter le
[wiki Analog Devices](https://wiki.analog.com/university/tools/pluto/users/firmware)
pour mettre à jour.

---

## Utilisation

### Mode matériel (PlutoSDR connecté)

```bash
python -m MicroDopplerDetection.main --config MicroDopplerDetection/config.yaml
```

### Mode simulation (sans matériel)

```bash
python -m MicroDopplerDetection.main --config MicroDopplerDetection/config.yaml --simulation
```

Le flag `--simulation` force `simulation.enable: true` quel que soit le contenu
du fichier de configuration.

### Options CLI

| Option          | Défaut                                  | Description                            |
|-----------------|-----------------------------------------|----------------------------------------|
| `--config`      | `MicroDopplerDetection/config.yaml`     | Chemin du fichier de configuration      |
| `--simulation`  | *(absent)*                              | Activer le mode simulation             |

---

## Paramètres (`config.yaml`)

Le fichier `config.yaml` est l'**unique source de vérité** pour tous les
paramètres du pipeline. Chaque section correspond à une étape :

| Section          | Paramètres clés                      | Effet physique                                                                 |
|------------------|--------------------------------------|--------------------------------------------------------------------------------|
| `sdr`            | `f_c`, `f_s`, `rx_gain`, `tx_gain`   | Longueur d'onde λ, bande passante, dynamique ADC                               |
| `emission`       | `mode`, `f_offset`                   | Position du signal utile dans le spectre (loin du DC si `cw_offset`)           |
| `decimation`     | `D`, `f_max_utile`                   | Taux d'échantillonnage effectif, respect du critère de Shannon                 |
| `clutter`        | `mode`, `alpha`                      | Suppression du retour statique dominant à 0 Hz                                 |
| `windowing`      | `mode`                               | Compromis résolution fréquentielle / fuite spectrale                           |
| `spectrogramme`  | `n_fft`, `overlap`                   | Résolution δf = f_s_dec / n_fft, lissage temporel                              |
| `detection`      | `bande_respiration`, `seuil_snr_dB`  | Fréquences de recherche, sensibilité de l'alerte                               |
| `affichage`      | `dynamique_dB`, `colormap`, `ylim`   | Rendu visuel du waterfall et des courbes                                       |
| `simulation`     | `fv`, `D_mm`, `snr_dB`              | Paramètres du signal respiratoire synthétique                                  |

Voir les commentaires détaillés directement dans `config.yaml`.

---

## Pièges courants

### 1. Saturation de l'ADC

**Symptômes** : signal IQ écrêté, harmoniques parasites dans le spectre.
**Cause** : `rx_gain` trop élevé ou `tx_gain` insuffisamment atténué.
**Correction** : réduire `rx_gain` ou rendre `tx_gain` plus négatif. Viser
max(|IQ|) entre 50 % et 80 % de la pleine échelle (1024–1638 sur 2048).

### 2. Pic DC (0 Hz) dominant

**Origine** : couplage direct TX→RX (isolation finie), offset DC de l'ADC,
réflexions sur les objets statiques environnants.
**Solutions** :
- Utiliser `emission.mode: "cw_offset"` pour décaler le signal utile.
- Activer la suppression de clutter (`clutter.mode: "iir"` ou `"mti"`).

### 3. Violation du critère de Shannon à la décimation

**Symptôme** : repliement spectral (aliasing), fausses raies.
**Cause** : facteur `D` trop grand → f_s_new < 2 × f_max_utile.
**Correction** : réduire `D` ou augmenter `f_s`. Le pipeline lève une
`ValueError` explicite si la condition n'est pas respectée.

### 4. Bruit de phase sur longues acquisitions

**Symptôme** : élargissement des raies spectrales au fil du temps.
**Cause** : instabilité de l'oscillateur local du PlutoSDR sur des durées
> 10 s.
**Atténuation** : acquisitions plus courtes, moyennage incohérent, ou
traitement par fenêtre glissante (déjà implémenté via la STFT).

### 5. Isolation TX/RX insuffisante

**Symptôme** : clutter résiduel très puissant même après filtrage.
**Cause** : le signal TX fuit directement dans le récepteur.
**Solutions matérielles** : antennes séparées avec écartement, circulateur,
absorbant entre TX et RX.
**Solutions logicielles** : `clutter.mode: "iir"` avec `alpha` élevé.

---

## Licence

Projet interne — usage académique et recherche.
