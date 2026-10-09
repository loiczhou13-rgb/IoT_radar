[English](README.md) | **Français**

# IoT Radar — détection de la respiration de personnes ensevelies

Un **radar à onde continue (CW)** portable, construit sur un SDR
**ADALM-Pluto** (AD9363), qui détecte la respiration de personnes ensevelies
sous des décombres (séisme, effondrement, avalanche).  Le mouvement du thorax
module la **phase** de l'écho ; le logiciel démodule cette phase, la convertit
en déplacement du thorax en millimètres et décide, toutes les demi-secondes,
si quelqu'un respire, retient sa respiration (apnée) ou bouge.

Projet étudiant (S6, équipe IoT).  Tout tourne en Python 3.10+, avec ou sans
le radar : une simulation physique et les sessions enregistrées alimentent
exactement la même chaîne de traitement.

> **État.** La chaîne de phase est validée sur le radar simulé (fréquence à
> ±1 respiration/min près, amplitude du déplacement, apnée, scène vide,
> discontinuités du flux — voir `tests/test_vital_signs_pipeline.py`).  Elle
> reste à valider sur le PlutoSDR : sur les anciens enregistrements, elle ne
> détecte pas la respiration, pour une raison encore inconnue (voir
> [Limites connues](#limites-connues)).

---

## Sommaire

1. [Inventaire](#inventaire)
2. [Principe](#principe)
3. [Organisation du dépôt](#organisation-du-dépôt)
4. [Installation](#installation)
5. [Démarrage rapide](#démarrage-rapide)
6. [Configuration](#configuration)
7. [Fichiers de session (HDF5)](#fichiers-de-session-hdf5)
8. [Tableau de bord](#tableau-de-bord)
9. [Tests](#tests)
10. [Règles d'organisation du code](#règles-dorganisation-du-code)
11. [Limites connues](#limites-connues)

---

## Inventaire

État : branche où le composant existe (`dev`, `feature/mimo-localization`, `siso-detection`).  Tests : couvert par `tests/` de cette branche.

### Acquisition et matériel

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Émission CW | `acquisition/pluto.py` (`cw_tx_buffer`) | buffer TX cyclique, décalage arrondi à un nombre entier de périodes | | dev | oui |
| Configuration du Pluto | `acquisition/pluto.py` (`open_pluto`) | pyadi-iio, 1 MS/s, 16 buffers noyau, émission cyclique | | dev | oui |
| Contrôles RX | `acquisition/pluto.py` | registre de pertes `0x80000088`, alerte de saturation à 80 % | | dev | oui |
| Source Pluto | `acquisition/sources.py` (`PlutoSource`) | blocs horodatés, signalement des pertes | | dev | oui |
| Interface des sources | `acquisition/sources.py` (`Block`, `Source`, `open_source`) | protocole commun Pluto / simulation / relecture | | dev | oui |
| Mode 2r2t | `acquisition/pluto.py` (`open_pluto_2r2t`) | `adi.ad9361`, 2 RX cohérents, 1 ou 2 TX, un ton par TX | | feature/mimo-localization, non testé sur matériel | oui |
| Nœud de localisation | `acquisition/sources.py` (`NodeSetup`, `open_node_source`) | nœud de `localization.yaml` : Pluto 2r2t ou scène simulée | | feature/mimo-localization, non testé sur matériel | oui |

### Simulation

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Radar CW simulé | `acquisition/sources.py` (`CWSimulationSource`) | fouillis à f_TX, DC du récepteur, dérive de LO, quantification ADC, chronologie | | dev | oui |
| Personne simulée | `acquisition/sources.py` (`SimulatedPerson`) | respiration, cœur, mouvement, apnée | | feature/mimo-localization | oui |
| Scène multi-nœuds | `acquisition/sources.py` (`SceneSimulationSource`) | n_tx tons, n_rx voies cohérentes, réflecteurs, tons de l'autre nœud | | feature/mimo-localization | oui |
| Perturbateurs | `acquisition/disturbers.py` | marcheur (tronc et membres), balancement, geste | Chen 2006 | siso-detection | oui |
| Masquage des trajets | `acquisition/disturbers.py` (`path_masking`) | jusqu'à −20 dB, transition en cosinus surélevé | | siso-detection | oui |
| Réflecteurs statiques | `acquisition/sources.py` (`static_scatterers`) | murs masqués par les perturbateurs | | siso-detection | oui |

### Données

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Sessions HDF5 v1 | `acquisition/recording.py` | IQ int16, blocs, annotations, vérité terrain, SWMR | | dev | oui |
| Sessions de nœud (v2) | `acquisition/recording.py` | un fichier par nœud, géométrie, positions de la cible | | feature/mimo-localization | oui |
| Intervalles de perturbation | `acquisition/recording.py` | `/ground_truth/disturbances` : début, durée, type | | siso-detection | oui |
| Relecture | `acquisition/sources.py` (`ReplaySource`) | blocs d'une session au rythme d'origine | | dev | oui |
| Conversion des anciens `.iq` | `scripts/convert_legacy_iq.py` | `.iq` + `.json` → HDF5 v1 | | dev | oui |
| Fichiers de calibration | `localization.py` (`save_calibration`, `load_calibration`) | YAML par nœud | | feature/mimo-localization | oui |

### Traitement du signal

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Mélangeur | `dsp/mixer.py` | NCO à phase continue, écho → 0 Hz | | dev | oui |
| Décimation | `dsp/decimation.py` | cascade de Tchebychev I d'ordre 8, facteurs premiers ≤ 13 | | dev | oui |
| Vue micro-Doppler | `dsp/spectral.py` | fenêtres, colonnes de TFCT (affichage) | | dev | oui |
| Filtres | `dsp/filters.py` | passe-bande de Butterworth à phase nulle, retrait de tendance | | dev | oui |
| Ajustement de cercle | `dsp/phase.py` (`fit_circle`) | Kåsa puis Gauss-Newton | Paterniani 2023 | dev | oui |
| Démodulation de phase | `dsp/phase.py` (`demodulate`) | arctangente, DACM ou linéaire, après dérotation de la dérive de LO | Paterniani 2023 | dev | oui |
| Filtrage Doppler (M1) | `dsp/doppler.py` | RIF de Kaiser à phase linéaire, −6 dB à 1,5 Hz, retard 1,82 s | Crochiere 1981 | siso-detection | oui |
| Audit du repliement | `dsp/doppler.py` (`alias_audit`) | atténuation des bandes repliées dans ±1 Hz, par étage et composite | Crochiere 1981 | siso-detection | oui |
| Centre glissant (M2) | `dsp/phase.py` (`track_center`, `demodulate_tracked`) | Pratt sur sous-fenêtres de 10 s, interpolation linéaire, désactivé par défaut | Pratt 1987, Zakrzewski 2012, Park 2007 | siso-detection | oui |
| Contamination (M3) | `dsp/contamination.py` | densité hors bande / dans la bande par bloc de 1 s, poids logistique, activité | VanderPlas 2018 | siso-detection | oui |
| Composante respiratoire multi-voies | `dsp/doa.py` (`breathing_component`, `tx_phase_reference`) | passe-bande commun, référence de phase par TX sur la fuite | | feature/mimo-localization | oui |

### Estimation et détection

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Estimateurs de fréquence | `dsp/estimation.py` | périodogramme, pic FFT, ACF, comptage de pics, passages par zéro | | dev | oui |
| Cycles respiratoires | `dsp/estimation.py` (`breath_cycles`) | inspirations, intervalles, profondeur, rapport I:E | | dev | oui |
| Métriques de détection | `dsp/detection.py` (`breathing_metrics`) | SNR de bande, concentration du pic, mouvement → confiance | | dev | oui |
| Détecteur | `dsp/detection.py` (`BreathingDetector`) | lissage, hystérésis ON/OFF, maintien de la pause, alerte d'apnée | | dev | oui |
| Pipeline de phase | `pipeline.py` (`VitalSignsPipeline`) | fenêtre de 20 s toutes les 0,5 s, redémarrage sur discontinuité | | dev | oui |
| Lomb-Scargle pondéré (M3) | `dsp/estimation.py` (`weighted_lomb_scargle`) | moyenne flottante, un décalage par segment | VanderPlas 2018 | siso-detection | oui |
| Règle de pause (M3) | `pipeline.py` | anomalies dans la bande coupées, imputées à un passant ou à la cible | | siso-detection | oui |
| Mode mesure (M4) | `measurement.py`, `pipeline.py` | 30 s sur un silence → décision, fréquence, résumé de qualité | | siso-detection | oui |
| Pipeline multi-voies | `pipeline.py` (`node_report`) | démodulation par ton, voies virtuelles, voie formée | | feature/mimo-localization | oui |

### Localisation

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Réseau virtuel | `dsp/doa.py` (`virtual_positions_m`, `steering_vectors`) | éléments tx_k + rx_l, champ lointain | | feature/mimo-localization | oui |
| Direction d'arrivée | `dsp/doa.py` (`estimate_doa`) | formation de voies de Bartlett, affinage parabolique, borne de Cramér-Rao | | feature/mimo-localization, non testé sur matériel | oui |
| Calibration des voies | `dsp/doa.py` (`estimate_calibration`), `scripts/calibrate_doa.py` | vecteur propre principal, cible à angle connu | | feature/mimo-localization, non testé sur matériel | oui |
| Géométrie des nœuds | `localization.py` (`NodeGeometry`) | pose, antennes, repères scène et nœud | | feature/mimo-localization | oui |
| Triangulation | `localization.py` (`triangulate`) | moindres carrés sur les angles puis Gauss-Newton, ellipse χ² | | feature/mimo-localization, non testé sur matériel | oui |
| Cohérence entre nœuds | `localization.py` (`check_consistency`) | écart de fréquence, corrélation des formes d'onde | | feature/mimo-localization | oui |
| Fusion | `localization.py` (`LocalizationFusion`) | dernier rapport par nœud, écart temporel maximal | | feature/mimo-localization | oui |
| Plan de fréquences | `localization.py` (`check_frequency_plan`) | écart des tons aux harmoniques, images et intermodulations | | feature/mimo-localization, non testé sur matériel | oui |
| Contrôle de la PIRE | `localization.py` (`estimated_eirp_dbm`) | refus d'émettre au-delà de 10 dBm | | feature/mimo-localization, non testé sur matériel | oui |

### Physique

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Bilan de liaison | `physics.py` (`radar_range_m`, `range_interval_m`) | équation radar monostatique, portées pessimiste et optimiste | | dev | oui |

### Apprentissage

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Jeu de données | `ml/dataset.py` | fenêtres de TFCT des anciens spectrogrammes `.npz` | | dev | oui |
| Autoencodeur | `ml/model.py` | Conv2D encodeur-décodeur + tête de classification | | dev | oui |
| Entraînement | `ml/train.py`, `scripts/train.py` | AdamW, StepLR, reconstruction + BCE | | dev | oui |

### Interface

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Tableau de bord de phase | `ui/dashboard.py` | matplotlib, sorties produites dans un thread | | dev | oui |
| Indicateurs M3 et mesure | `ui/dashboard.py` | poids, part contaminée, cause de pause, touche `m` | | siso-detection | oui |
| Écran d'accueil | `ui/launcher.py`, `scripts/launcher.py` | pygame, radar lancé en sous-processus | | dev | oui |
| Tableau de bord de localisation | `ui/localization_dashboard.py` | vue de dessus, relèvements, ellipse, spectres P(θ) | | feature/mimo-localization | oui |

### Outillage

| Composant | Emplacement | Méthode | Référence | État | Tests |
|---|---|---|---|---|---|
| Radar en direct | `scripts/run_radar.py` | source → pipeline → tableau de bord ou journal | | dev | oui |
| Mesure sans écran | `scripts/run_radar.py` (`--measure`) | une mesure M4 puis sortie | | siso-detection | oui |
| Enregistrement | `scripts/record.py` | sessions IQ brutes, label, scène | | dev, non testé sur matériel | oui |
| Relecture | `scripts/replay.py` | session → pipeline → tableau de bord | | dev | oui |
| Localisation en direct | `scripts/run_localization.py` | un processus par nœud, fusion, affichage | | feature/mimo-localization, non testé sur matériel | oui |
| Contrôle matériel 2RX | `scripts/hardware_check_2rx.py` | 2r2t, puis 1T2R et 2T2R : tons et phases | | feature/mimo-localization, non testé sur matériel | oui |
| Évaluation de la robustesse | `scripts/evaluate_motion_robustness.py` | scènes S1–S8, sessions, Pd/Pfa, balayage de κ₀ | | siso-detection | non |
| Configuration | `config.py` | YAML, chemins du dépôt, journalisation | | dev | oui |
| Références de caractérisation | `tests/characterization_cases.py`, `tests/make_references.py` | sorties DSP comparées au bit près | | dev | oui |
| Règles de dépendance | `tests/test_architecture.py` | imports interdits entre couches | | dev | oui |
| Faux Pluto | `tests/fake_pluto.py` | pyadi-iio simulé en 2r2t | | feature/mimo-localization | oui |

### Points d'attention

- Détection jamais validée sur données réelles : 2 % d'analyses `BREATHING` sur les 69 anciennes sessions, cause inconnue (pertes d'échantillons de l'ancien code à 2 MS/s suspectées).
- Radar complet (TX + RX + tableau de bord) à 1 MS/s sans perte encore à confirmer ; le tableau de bord redessine dans le processus d'acquisition, 2 MS/s impossible.
- Porteuse : `radar.yaml` à 3,5 GHz sur `dev` (bande des opérateurs mobiles en France, non ISM) ; 2,45 GHz sur `siso-detection`, 2,42 GHz pour les nœuds de `feature/mimo-localization` : à harmoniser.
- Localisation : chaîne matérielle jamais vérifiée (Pluto rév. C/D et U.FL, mode 2r2t par SSH, stabilité des phases RX1/RX2 et entre TX, 600 kS/s × 2 voies sans perte).
- PIRE calculée avec P(0 dB) ≈ +7 dBm et des antennes de 6 dBi supposées : à mesurer (limite 10 mW).
- DoA : hypothèse d'une fuite TX→RX dominant le fouillis ; biais de la référence statique (~0,3°) non corrigé ; une seule cible ; trajets multiples non traités.
- Seuils de M3 (κ₀ 9 dB, activité 6 dB, excursions 12 mm, segments 5 s) réglés sur simulation seulement ; perturbateurs annotés dans les seules sessions simulées.
- Une seule antenne : un perturbateur proche bougeant dans la bande respiratoire (balancement à 2 m) reste inséparable de la cible.
- M3 sans M1 donne de fausses alarmes (Pfa 12 % en simulation) : garder M1 dès que M3 est activé.
- Fusion des deux branches : toutes deux modifient `sources.py`, `pipeline.py`, `recording.py`, `dashboard.py` et `tests/data/references.npz` (binaire) → conflits, références à régénérer.
- `ml/` obsolète et non branché : il apprend sur les `.npz` de l'ancienne chaîne micro-Doppler, qui ne sont plus produits.
- `data/` est ignoré par Git : les 69 sessions converties n'existent que sur une machine.
- Écarts plan/code : `MOTION_ROBUSTNESS_PLAN.md` annonce une classe `DopplerLowpass` (code : fonction `doppler_lowpass_decimator`) ; B9 de `REFACTOR_PLAN.md` (débit `ip:` à 2 MS/s) est caduc depuis le passage à 1 MS/s.
- FMCW sur Pluto écarté : USB 2.0 insuffisant pour 20 MS/s, résolution de 8,3 m pour 18 MHz de bande, cibles masquées par la fuite TX→RX.
- `ed_branch` modifie des fichiers supprimés par le refactor (`MicroDopplerDetection/`) : conflit en cas de fusion.

---

## Principe

### Physique

Le Pluto émet une onde continue à `f_c = 3,5 GHz` (longueur d'onde
`λ = c / f_c = 8,57 cm`).  Un thorax qui se déplace de `d(t)` vers le radar
décale la phase de son écho de

```
φ(t) = −4π · d(t) / λ
```

Une respiration de 10 mm donne 1,47 rad ; un battement cardiaque de 0,1 mm
donne 0,015 rad.  Dans le plan IQ, les échantillons en temps lent sont sur un
**arc de cercle** centré sur la composante continue statique (fuite TX→RX,
murs, DC du récepteur).  Le déplacement se lit dans l'angle **autour de ce
centre** : c'est pourquoi la chaîne ajuste le cercle au lieu de filtrer l'IQ
par un passe-haut (bug B3 de l'ancienne chaîne).

En mode `cw_offset` (par défaut), la tonalité est émise à `f_c + 503,54 Hz`,
donc l'écho arrive loin du DC du récepteur et de son bruit en 1/f.  Le
décalage est arrondi pour que le buffer TX contienne un nombre entier de
périodes (500 Hz demandés → 503,54 Hz émis avec des buffers de 65536
échantillons à 1 MS/s, bug B1).

### Chaîne de traitement

```
PlutoSDR │ simulation │ relecture HDF5      iot_radar/acquisition/sources.py
        │  Block : samples (n_channels, n), sample_start, host_time_s, overflow
        ▼
Mixer (NCO à −tx_offset_hz)                dsp/mixer.py       écho → 0 Hz
        ▼
Decimator (cascade de Tchebychev)          dsp/decimation.py  1 MS/s → 20 Hz (temps lent)
        ▼
fenêtre glissante : 20 s, analysée toutes les 0,5 s   pipeline.py
        ▼
VitalSignsProcessor                        pipeline.py
   dérotation de la dérive de LO → ajustement du cercle (DC)   dsp/phase.py
   → arctangente / DACM → déplacement
   → passe-bande 0,1–0,5 Hz → spectre      dsp/filters.py, dsp/estimation.py
   → métriques de respiration (SNR,        dsp/detection.py
     concentration du pic, mouvement)
        ▼
BreathingDetector                          dsp/detection.py
   lissage, hystérésis ON/OFF, maintien MOTION, fréquence médiane, alerte d'apnée
        ▼
PipelineOutput (dataclasses) ──► tableau de bord (ui/dashboard.py) ou journal
```

Chaque bloc porte l'indice de son premier échantillon, l'heure de l'hôte et
un indicateur de débordement (*overflow*).  Quand des échantillons sont
perdus (débordement, ou trou dans `sample_start`), le pipeline réinitialise
l'oscillateur, les filtres, la fenêtre et le détecteur, marque sa sortie
suivante avec `discontinuity=True` et repart ; il ne recolle jamais deux
morceaux de signal.

États du détecteur : `WARMUP`, `NO_BREATHING`, `BREATHING`, `MOTION`.
L'alerte d'apnée est levée quand aucune inspiration n'a été vue depuis 10 s
alors qu'une respiration était détectée peu avant (un radar ne distingue pas
une apnée d'une personne qui sort du faisceau).

---

## Organisation du dépôt

```
IoT_radar/
├── iot_radar/                 paquet Python installable
│   ├── config.py              chemins du dépôt, lecture du YAML, journalisation
│   ├── physics.py             constantes physiques, équation radar
│   ├── acquisition/
│   │   ├── pluto.py           configuration du PlutoSDR, buffer TX, contrôles de débordement / saturation
│   │   ├── sources.py         Block, PlutoSource, CWSimulationSource, ReplaySource
│   │   └── recording.py       écriture et lecture des sessions HDF5 (schéma v1)
│   ├── dsp/                   traitement du signal (ni E/S, ni affichage)
│   │   ├── mixer.py           translation de fréquence (NCO)
│   │   ├── decimation.py      décimateur à état
│   │   ├── phase.py           ajustement du cercle, arctangente / DACM, phase → déplacement
│   │   ├── filters.py         passe-bande, suppression de tendance
│   │   ├── estimation.py      périodogramme, estimateurs de fréquence, cycles respiratoires
│   │   ├── detection.py       métriques de respiration et BreathingDetector
│   │   └── spectral.py        fenêtres et TFCT (vue micro-Doppler uniquement)
│   ├── pipeline.py            VitalSignsPipeline : blocs en entrée, PipelineOutput en sortie
│   ├── ui/
│   │   ├── dashboard.py       tableau de bord de phase temps réel (matplotlib)
│   │   └── launcher.py        écran d'accueil pygame
│   └── ml/                    ancien autoencodeur de spectrogrammes (voir son README)
├── scripts/                   points d'entrée en ligne de commande (voir Démarrage rapide)
├── configs/                   radar.yaml, training.yaml
├── notebooks/                 inference.ipynb (ml/)
├── tests/                     tests pytest, tests/data/references.npz
├── data/sessions/             sessions HDF5 (ignorées par Git)
├── logs/                      journaux d'exécution (ignorés par Git)
├── pyproject.toml, requirements.txt
└── REFACTOR_PLAN.md           décisions de conception du refactor de 2026
```

---

## Installation

### Système (PlutoSDR uniquement)

```bash
sudo apt update
sudo apt install -y libiio-dev libiio-utils python3-venv
```

```bash
iio_info -s
```

`iio_info -s` doit lister le Pluto.  Sous WSL2, attachez d'abord le
périphérique USB avec `usbipd`, ou utilisez l'adresse réseau
`ip:192.168.2.1`.

### Python

Depuis la racine du dépôt :

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

`requirements.txt` fixe les versions testées et installe le paquet en mode
éditable (`pip install -e .`).  Pour une installation plus légère, choisissez
les extras utiles :

```bash
pip install -e ".[hardware,ui,dev]"
```

| Extra | Paquets | Nécessaire pour |
|---|---|---|
| *(base)* | numpy, scipy, matplotlib, pyyaml, h5py | traitement, tableau de bord, sessions |
| `hardware` | pyadi-iio, pylibiio | le PlutoSDR |
| `ui` | pygame | l'écran d'accueil |
| `ml` | torch, tqdm | `scripts/train.py` |
| `notebook` | ipykernel | `notebooks/inference.ipynb` |
| `dev` | pytest | les tests |

---

## Démarrage rapide

Toutes les commandes se lancent depuis la racine du dépôt, environnement
virtuel activé.

**Radar en simulation** (sans matériel) :

```bash
python scripts/run_radar.py --simulation
```

**Radar sur le PlutoSDR :**

```bash
python scripts/run_radar.py
```

Options : `--config FICHIER`, `--headless` (pas de fenêtre, chaque résultat
va dans le journal), `--log-file FICHIER`.  Les journaux vont par défaut dans
`logs/radar_<horodatage>.log`.

**Écran d'accueil**, avec un bouton par mode (chacun lance `run_radar.py`
dans son propre processus) :

```bash
python scripts/launcher.py
```

**Enregistrer une session** (IQ brut, sans affichage).  Deux minutes d'une
personne qui respire à 2,5 m derrière 20 cm de béton :

```bash
python scripts/record.py --label breathing --duration-s 120 --subject-id S01 --distance-m 2.5 --obstacle concrete --obstacle-thickness-cm 20 --room lab_b12
```

`-n 5 --interval-s 120` enregistre cinq sessions avec une pause de deux
minutes après chacune.  Avec `--simulation`, le label fixe la scène simulée,
et la session stocke aussi le déplacement simulé du thorax comme vérité
terrain.

**Relire une session** dans le pipeline et le tableau de bord (configuration
stockée dans la session, `--speed 0` = aussi vite que possible) :

```bash
python scripts/replay.py data/sessions/session_0001_20261005_143012.h5
```

**Convertir les anciens enregistrements `.iq`** en sessions HDF5
(`data/sessions/legacy/` par défaut) :

```bash
python scripts/convert_legacy_iq.py chemin/vers/anciennes/donnees
```

**Entraîner l'ancien autoencodeur** (voir [`iot_radar/ml/README.fr.md`](iot_radar/ml/README.fr.md)) :

```bash
python scripts/train.py --epochs 100
```

---

## Configuration

[`configs/radar.yaml`](configs/radar.yaml) est l'unique fichier de réglages
du radar ; chaque clé est commentée (en anglais) et son suffixe donne son
unité (`_hz`, `_s`, `_m`, `_mm`, `_db`, …).

| Section | Rôle |
|---|---|
| `logging` | niveau de journalisation, fichier de log activé ou non |
| `sdr` | adresse du Pluto, porteuse, fréquence d'échantillonnage, gains, taille de buffer |
| `tx` | forme d'onde (`cw` / `cw_offset`) et décalage |
| `slow_time` | fréquence du temps lent après décimation (20 Hz), préchauffage |
| `vital_signs` | fenêtre d'analyse, bandes, compensation du DC, démodulation, validité de l'ajustement du cercle, seuils de `detection` |
| `micro_doppler_view` | waterfall du tableau de bord (affichage uniquement) |
| `display` | durée d'historique du tableau de bord, plein écran |
| `link_budget` | scénarios optimiste / pessimiste → portée affichée |
| `recording` | dossier des sessions, intervalle d'écriture sur disque |
| `simulation` | scène simulée : présence, chronologie, respiration, cœur, mouvement, fouillis, SNR, graine |

Une scène simulée peut évoluer dans le temps, par exemple :

```yaml
simulation:
  enabled: true
  timeline: [[0, breathing], [45, apnea], [60, breathing], [95, motion], [110, empty]]
```

**Fréquence d'échantillonnage.**  Le Pluto échantillonne à 1 MS/s, avec des
buffers de 65536 échantillons et 16 buffers RX mis en file dans le Pluto
(`sdr.rx_kernel_buffers`, 1 s de marge).  À 2 MS/s en USB, le PC ne suivait
pas pendant les redessins du tableau de bord (même processus Python) : le
Pluto perdait des échantillons et le pipeline redémarrait sans cesse.  Les
pertes sont écrites dans le journal (« RX samples lost ») et signalées sur le
tableau de bord comme discontinuités du flux.  Si elles reviennent (PC plus
lent, autre liaison USB), baissez `sample_rate_hz` ou augmentez
`rx_kernel_buffers`.

[`configs/training.yaml`](configs/training.yaml) configure `ml/`.

---

## Fichiers de session (HDF5)

Un fichier par session, écrit par `scripts/record.py`
([`acquisition/recording.py`](iot_radar/acquisition/recording.py)) et jamais
modifié après l'acquisition.  Nom :
`session_<id>_<AAAAMMJJ>_<HHMMSS>.h5` (heure de début en UTC) ; la scène est
dans les métadonnées, pas dans le nom.

```
/                 attributs (ci-dessous)
/iq               int16 (n_channels, n_samples, 2) : I et Q en unités ADC
                  (fois son attribut "scale"), découpé en blocs le long du temps
/blocks           (sample_start, host_time_s, overflow) : une ligne par bloc reçu
/annotations      (sample_start, sample_count, label)
/ground_truth/    optionnel (simulation) : time_s, chest_displacement_m
```

| Attributs | |
|---|---|
| radar | `schema_version`, `sample_rate_hz`, `center_frequency_hz`, `tx_waveform`, `tx_offset_hz`, `tx_gain_db`, `rx_gain_db`, `rx_gain_mode`, `n_rx_channels`, `channel_layout`, `firmware_version`, `source_kind`, `iq_stage` |
| session | `start_time_utc`, `software_version`, `config_yaml` (configuration complète utilisée) |
| scène | `subject_id`, `distance_m`, `orientation`, `obstacle`, `obstacle_thickness_cm`, `room`, `notes` |

Les valeurs inconnues sont stockées comme `""` ou `NaN`.

Labels : `empty` (personne dans le champ), `breathing` (une personne,
immobile, qui respire), `apnea` (une personne, immobile, qui ne respire pas),
`motion` (les mouvements du corps dominent), `unknown` (non annoté ou
incertain).

Le fichier est écrit en mode SWMR et vidé sur disque toutes les secondes : si
le programme est tué, tout ce qui précède la dernière écriture reste lisible.
Lecture en Python :

```python
from iot_radar.acquisition.recording import SessionReader

with SessionReader("data/sessions/session_0001_20261005_143012.h5") as reader:
    iq = reader.read_iq(0, reader.n_samples)   # complex64 (n_channels, n_samples), unités ADC
    print(reader.sample_rate_hz, reader.attributes["distance_m"], reader.annotations())
```

---

## Tableau de bord

`scripts/run_radar.py` et `scripts/replay.py` ouvrent le tableau de bord de
phase ([`ui/dashboard.py`](iot_radar/ui/dashboard.py)) :

- **carte d'état** : état, fréquence respiratoire, jauge de confiance,
  bandeau d'apnée, discontinuités du flux ;
- **forme d'onde du déplacement** (brute et filtrée, inspiration vers le
  haut) avec les respirations détectées ;
- **constellation IQ** de la fenêtre analysée avec le cercle ajusté et son
  centre — montre si la phase peut être extraite ;
- **chronologie de la confiance**, colorée par état ;
- **grandeurs mesurées** : fréquences de quatre estimateurs, intervalle et
  variabilité des respirations, profondeur, rapport I:E, SNR, amplitudes de
  l'écho et du DC, dérive de LO, fréquence cardiaque indicative ;
- **historique de la fréquence**, **spectre du déplacement** et **waterfall
  micro-Doppler** du temps lent ;
- **paramètres** du radar, du traitement et de la détection.

Le pipeline ne connaît pas le tableau de bord : il renvoie des dataclasses
`PipelineOutput` que le tableau de bord dessine.

---

## Tests

```bash
pytest
```

La suite (moins d'une minute, sans matériel) couvre :

- les briques DSP ;
- les sources, y compris un faux Pluto ;
- les sessions HDF5 et l'équivalence bit à bit entre direct et relecture ;
- la validation de toute la chaîne sur le radar simulé ;
- les scripts ;
- des tests de fumée sans écran du tableau de bord et de l'écran d'accueil ;
- les règles de dépendance ci-dessous.

`tests/test_characterization.py` compare bit à bit des sorties DSP avec
`tests/data/references.npz`.  Ne régénérez ces références
(`python tests/make_references.py`) que si un changement des résultats
numériques est voulu, dans un commit dédié qui explique pourquoi.

---

## Règles d'organisation du code

Vérifiées par `tests/test_architecture.py` :

| Module | Ne doit pas importer |
|---|---|
| `dsp/` | `acquisition`, `ui`, `ml`, `pipeline` |
| `pipeline.py` | `ui`, `ml` (`acquisition` seulement pour les annotations de type) |
| `ui/` | `acquisition` (même indirectement), `ml` |
| `ml/` | `acquisition` |
| `config.py`, `physics.py` | rien du paquet |

Les scripts assemblent `acquisition` → `pipeline` → `ui`.  Conventions :
anglais partout dans le code, noms physiques avec un suffixe d'unité (`_hz`,
`_s`, `_m`, `_db`, `_rad`), docstrings avec formes et unités sur les fonctions
DSP.

---

## Limites connues

- **Validation matérielle en cours.**  La détection des échantillons perdus
  (registre `0x80000088`) fonctionne sur le Pluto : elle a révélé les pertes
  à 2 MS/s.  À 1 MS/s, un test en réception seule avec redessins du tableau
  de bord n'a perdu aucun échantillon ; reste à le confirmer avec le radar
  complet (TX + RX + tableau de bord), puis sur de nouveaux enregistrements.
- **Anciens enregistrements.**  Les 69 anciens fichiers `.iq` sont convertis
  dans `data/sessions/legacy/` (`iq_stage = "decimated_clutter_filtered"`).
  Leur filtre passe-haut n'agissait qu'autour de 0 Hz, alors que l'écho est
  à +488 Hz : la phase peut donc être retraitée.  Pourtant, la chaîne de
  phase **ne détecte pas** la respiration sur ces enregistrements :
  - sur les sessions `breathing`, seules 2 % des analyses sont en
    `BREATHING` et 57 % en `MOTION` (plus de 30 mm crête à crête) ;
  - sur les sessions `empty`, 12 % des analyses sont en `BREATHING`.

  La cause n'est pas établie : sujets qui bougent, phase trop bruitée (le
  résidu de l'ajustement du cercle est proche de celui d'un nuage de bruit),
  ou **échantillons perdus**.  L'ancien code lisait le Pluto à 2 MS/s sans
  vérifier les pertes, et chaque perte fait sauter la phase.  Dans la seule
  session vide assez propre pour le vérifier, la phase reste à 0,007 rad près
  entre des paquets de sauts allant jusqu'à 3 rad, mais quelqu'un qui bouge
  dans la pièce donnerait la même signature.  Il faut de nouveaux
  enregistrements bruts d'une scène contrôlée (`scripts/record.py`).  Détails
  au §19.3 de `REFACTOR_PLAN.md`.
- **`ml/`** apprend encore sur les anciens spectrogrammes `.npz` de la chaîne
  micro-Doppler supprimée ; il sera repensé sur le signal de phase des
  sessions HDF5.
- **FMCW** (respiration avec sélection en distance) a été étudié mais ne fait
  pas partie de ce paquet ; voir la section « FMCW: findings » de
  `REFACTOR_PLAN.md`.

---

## Licence

Projet interne — usage académique et recherche.
