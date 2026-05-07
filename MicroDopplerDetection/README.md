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

## Architecture du pipeline (streaming)

```
┌─────────────┐      ┌────────────────┐      ┌──────────────┐
│ emission.py │─────▶│ acquisition.py │─────▶│decimation.py │
│  buffer TX  │      │  IQ (PlutoSDR  │      │  Decimator   │
│  ×2¹⁴ scale │      │  ou simulation)│      │  stateful    │
└─────────────┘      └────────────────┘      └──────┬───────┘
                                                    │
                                            ┌───────▼──────┐
                                            │  clutter.py  │
                                            │ ClutterFilter│
                                            └──────┬───────┘
                                                   │
                                            ┌──────▼───────┐
                                            │spectrogramme │
                                            │     .py      │
                                            │compute_single│
                                            │   _column    │
                                            └──────┬───────┘
                                                   │
                                       ┌───────────▼────────────┐
                                       │     detection.py        │
                                       │ Fisher F-test  +  ACF   │
                                       │ → score & alerte        │
                                       └───────────┬─────────────┘
                                                   │
                                       ┌───────────▼─────────────┐
                                       │    utils/display.py     │
                                       │  Dashboard matplotlib   │
                                       │  3 panneaux + encadré   │
                                       └─────────────────────────┘
```

Orchestration dans **`main.py`** :
émission → acquisition → décimation streaming → clutter → fenêtrage + STFT
colonne par colonne → détection Fisher × ACF → affichage temps réel.

Le code historique en mode batch (acquisition complète, spectrogramme 2-D,
détection sur matrice) est préservé sous **`legacy/`** pour l'analyse
hors-ligne ; il n'est plus appelé par le pipeline.

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
| `scipy`       | Décimation IIR, fenêtrage, STFT, ACF               |
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

Le fichier `config.yaml` est l'**unique source de vérité** pour le pipeline.

| Section          | Paramètres clés                       | Effet                                                                          |
|------------------|----------------------------------------|--------------------------------------------------------------------------------|
| `sdr`            | `f_c`, `f_s`, `rx_gain`, `tx_gain`    | Longueur d'onde, bande, dynamique ADC                                          |
| `emission`       | `mode`, `f_offset`                    | Position du signal utile (loin du DC si `cw_offset`)                          |
| `decimation`     | `D`, `f_max_utile`                    | Taux d'échantillonnage effectif, vérif Shannon (avec `f_offset` pris en compte) |
| `clutter`        | `mode`, `alpha`, `butterworth_*`      | Suppression du retour statique à 0 Hz                                          |
| `windowing`      | `mode`                                | Compromis résolution / fuite spectrale                                         |
| `spectrogramme`  | `n_fft`, `overlap`, `skip_warmup`     | Résolution δf, lissage, warm-up clutter (auto si `null`)                       |
| `detection`      | `bande_respiration`, `bande_reference`, `alpha`, `w` | Test Fisher F + ACF fusionnés                                                  |
| `affichage`      | `N_historique`, `seuil_proba`         | Score history, seuil visuel                                                    |
| `bilan_liaison`  | `optimiste`/`pessimiste`, `B_eff_hz`  | Portée min/max affichée                                                        |
| `simulation`     | `fv`, `D_mm`, `snr_dB`, `clutter_amplitude` | Paramètres du signal respiratoire synthétique                                  |

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
- Activer la suppression de clutter (par défaut `clutter.mode: "butterworth"`,
  alternatives `"iir"` ou `"mean"`).

### 3. TX absent du spectre alors que le pipeline tourne

**Origine** : oubli du facteur `2**14` (échelle DAC PlutoSDR).
**Correction** : déjà appliquée dans `pipeline/emission.py`. Si vous
modifiez le buffer TX, conserver la mise à l'échelle ; sans elle, le DAC
ne reçoit que ~1 LSB et n'émet rien d'observable à l'analyseur.

### 4. Violation du critère de Shannon à la décimation

**Symptôme** : repliement spectral, fausses raies.
**Cause** : `D` trop grand → `f_s_new < 2.5 × max(f_max_utile, |f_offset| + bande_ref_max)`.
**Correction** : réduire `D`, augmenter `f_s`, ou ajuster `f_max_utile`.
Le pipeline lève une `ValueError` explicite si la condition n'est pas
respectée.

### 5. Bruit de phase sur longues acquisitions

**Symptôme** : élargissement des raies spectrales au fil du temps.
**Cause** : instabilité de l'oscillateur local du PlutoSDR (> ~10 s).
**Atténuation** : la STFT en fenêtre glissante limite déjà l'effet ;
sinon, raccourcir `n_fft` ou augmenter le recouvrement.

### 6. Isolation TX/RX insuffisante

**Symptôme** : clutter résiduel très puissant même après filtrage.
**Cause** : le signal TX fuit directement dans le récepteur.
**Solutions matérielles** : antennes séparées, circulateur, absorbant
entre TX et RX.
**Solutions logicielles** : `clutter.mode: "iir"` avec `alpha` élevé,
ou `"butterworth"` avec coupure plus haute.

---

## Licence

Projet interne — usage académique et recherche.
