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
> discontinuités du flux — voir `tests/test_vital_signs_pipeline.py`), y
> compris avec des personnes qui bougent autour du radar ([Robustesse aux
> mouvements parasites](#robustesse-aux-mouvements-parasites)).  Elle reste à
> valider sur le PlutoSDR : sur les anciens enregistrements, elle ne détecte
> pas la respiration, pour une raison encore inconnue (voir
> [Limites connues](#limites-connues)).

---

## Sommaire

1. [Inventaire](#inventaire)
2. [Principe](#principe)
3. [Robustesse aux mouvements parasites](#robustesse-aux-mouvements-parasites)
4. [Sur le terrain : mesure pendant un silence](#sur-le-terrain--mesure-pendant-un-silence)
5. [Organisation du dépôt](#organisation-du-dépôt)
6. [Installation](#installation)
7. [Démarrage rapide](#démarrage-rapide)
8. [Configuration](#configuration)
9. [Fichiers de session (HDF5)](#fichiers-de-session-hdf5)
10. [Tableau de bord](#tableau-de-bord)
11. [Tests](#tests)
12. [Règles d'organisation du code](#règles-dorganisation-du-code)
13. [Limites connues](#limites-connues)

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
- Écarts plan/code : `MOTION_ROBUSTNESS_PLAN.md` annonce une classe `DopplerLowpass` (code : fonction `doppler_lowpass_decimator`).
- FMCW sur Pluto écarté : USB 2.0 insuffisant pour 20 MS/s, résolution de 8,3 m pour 18 MHz de bande, cibles masquées par la fuite TX→RX.
- `ed_branch` modifie des fichiers supprimés par le refactor (`MicroDopplerDetection/`) : conflit en cas de fusion.

---

## Principe

### Physique

Le Pluto émet une onde continue à `f_c = 2,45 GHz`, dans la bande ISM
(longueur d'onde `λ = c / f_c = 12,24 cm`).  Un thorax qui se déplace de `d(t)` vers le radar
décale la phase de son écho de

```
φ(t) = −4π · d(t) / λ
```

Une respiration de 10 mm donne 1,03 rad ; un battement cardiaque de 0,1 mm
donne 0,010 rad.  Dans le plan IQ, les échantillons en temps lent sont sur un
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
Decimator (cascade de Tchebychev)          dsp/decimation.py  1 MS/s → 100 Hz (débit Doppler)
        ├──► M3 contamination par bloc de 1 s   dsp/contamination.py  poids, activité
        ▼
M1 passe-bas Doppler (RIF), ×5             dsp/doppler.py     100 Hz → 20 Hz (temps lent)
        ▼
fenêtre glissante : 20 s, analysée toutes les 0,5 s   pipeline.py
        ▼
VitalSignsProcessor                        pipeline.py
   dérotation de la dérive de LO → ajustement du cercle (DC),   dsp/phase.py
     (M2, optionnel : centre glissant s'il dérive)
   → arctangente / DACM → déplacement
   → M3 anomalies dans la bande coupées (poids 0)
   → passe-bande 0,1–0,5 Hz → spectre      dsp/filters.py, dsp/estimation.py
     (M3 : Lomb-Scargle pondéré)
   → métriques de respiration (SNR,        dsp/detection.py
     concentration du pic, mouvement)
        ▼
BreathingDetector                          dsp/detection.py
   lissage, hystérésis ON/OFF, maintien de la pause, fréquence médiane, alerte d'apnée
        ▼
M4 mode mesure (30 s, à la demande)        measurement.py
        ▼
PipelineOutput (dataclasses) ──► tableau de bord (ui/dashboard.py) ou journal
```

Chaque bloc porte l'indice de son premier échantillon, l'heure de l'hôte et
un indicateur de débordement (*overflow*).  Quand des échantillons sont
perdus (débordement, ou trou dans `sample_start`), le pipeline réinitialise
l'oscillateur, les filtres, la fenêtre et le détecteur, marque sa sortie
suivante avec `discontinuity=True` et repart ; il ne recolle jamais deux
morceaux de signal.

États du détecteur : `WARMUP`, `NO_BREATHING`, `BREATHING`, `MOTION`
(mesure en pause ; `pause_reason` en donne la cause : `motion`, mouvement
de la cible, ou `saturation` de l'ADC).  L'alerte d'apnée est levée quand
aucune inspiration n'a été vue depuis 10 s alors qu'une respiration était
détectée peu avant (un radar ne distingue pas une apnée d'une personne qui
sort du faisceau).

---

## Robustesse aux mouvements parasites

En SISO, toute activité dans le faisceau mettait la mesure en pause : en
milieu ouvert, le radar était presque toujours en pause.  Sur le terrain, la
recherche se fait pendant des silences, mais il reste des mouvements
résiduels (personnes plus loin, balancement, gestes).  Ils sont désormais
**supprimés par traitement au lieu de mettre en pause** (conception et
choix : [`MOTION_ROBUSTNESS_PLAN.md`](MOTION_ROBUSTNESS_PLAN.md)).

Une vitesse radiale `v` donne le Doppler `f_D = 2·v/λ ≈ 16·v Hz` à 2,45 GHz :

| Source | Doppler à 2,45 GHz |
|---|---|
| respiration | ~0,15 Hz (harmoniques jusqu'à ~1 Hz) |
| balancement d'une personne debout | ~0,8 Hz |
| gestes | 3–16 Hz |
| marche | 16–25 Hz |

Les échos s'additionnent **linéairement** dans l'IQ, alors que l'arctangente
n'est pas linéaire : tout est supprimé sur l'IQ complexe, **avant** la
démodulation de phase.

- **M1 — filtrage Doppler** (`dsp/doppler.py`).  Passe-bas RIF à phase
  linéaire sur l'IQ à 100 Hz : plat jusqu'à 1 Hz, −6 dB à 1,5 Hz, ≥ 60 dB dès
  2 Hz, causal, retard constant de 1,82 s.  C'est aussi le filtre
  anti-repliement de la dernière décimation.  Le repliement de toute la
  chaîne dans ±1 Hz est audité au démarrage (≥ 84 dB, exigence 60 dB) : un
  marcheur à 19 Hz n'apparaît jamais comme une respiration.
- **M2 — centre glissant de l'arc** (`dsp/phase.py`).  Quand des personnes
  bougent, les réflexions statiques, donc le centre de l'arc IQ, dérivent.
  Ajustements de Pratt sur des sous-fenêtres de 10 s toutes les 1 s,
  interpolés linéairement (la phase reste continue).  Le centre suivi n'est
  utilisé que s'il s'ajuste nettement mieux qu'un centre fixe.
  **Désactivé par défaut** : il n'a apporté aucun gain mesurable à
  l'évaluation, où un passant ne masque un mur que brièvement (M3 coupe ce
  passage) ; l'activer là où le fouillis statique dérive lentement.
- **M3 — pondération et estimation tolérante aux trous**
  (`dsp/contamination.py`, `pipeline.py`).  Par bloc de 1 s, avant M1,
  l'indice de contamination (densité d'énergie au-dessus de 2 Hz sur densité
  dans la bande respiratoire) donne un poids lisse dans [0, 1].  Le spectre
  est un périodogramme de Lomb-Scargle pondéré : les passages contaminés sont
  sautés, et la confiance affichée est multipliée par le poids moyen.  La
  mesure ne se met en pause **que** pour une saturation de l'ADC ou un
  mouvement de la cible elle-même.
  - Un marcheur qui traverse le faisceau passe par une vitesse radiale
    nulle : pendant environ 1 s, son écho est dans la bande respiratoire.
    Cette anomalie dans la bande est coupée (poids 0).  Elle est imputée à
    un passant si quelqu'un bougeait hors bande (3–28 Hz au-dessus du bruit
    du récepteur) dans les 10 s précédentes : une personne n'atteint la
    bande qu'en ralentissant, on la voit donc venir.  Sinon, c'est la cible
    qui bouge, et la mesure se met en pause.
- **M4 — mode mesure** (`measurement.py`) : voir la section suivante.

M1 et M3 sont activés par défaut, M2 est désactivé.  Chacune peut être
activée ou désactivée dans la section `motion_robustness` ; les trois
désactivées, on retrouve exactement l'ancien comportement.

**Évaluation** ([`MOTION_ROBUSTNESS_EVAL.md`](MOTION_ROBUSTNESS_EVAL.md) ;
scènes simulées et 69 anciennes sessions).  Avec M1 + M3, par rapport à
l'ancienne chaîne :
- le temps en pause avec des personnes qui bougent autour passe de 34–100 %
  à 0–1 % ;
- mesures de 30 s : Pd de 58 % à 74 % (marcheurs qui coupent le faisceau à
  3 m : de 5 % à 70 %), Pfa de 5 % à 0 % ;
- un mouvement de la cible met toujours en pause, et ses mesures restent non
  concluantes.

---

## Sur le terrain : mesure pendant un silence

1. **Les sauveteurs sortent du faisceau et restent immobiles** : derrière le
   radar ou à plus de 60° sur le côté, personne à moins de ~3 m devant lui.
   Une personne proche du faisceau dont les mouvements restent dans la bande
   respiratoire (balancement, report du poids d'une jambe sur l'autre) ne
   peut pas être distinguée de la victime avec une seule antenne.
2. **Lancer la mesure quand le silence commence** : touche `m` du tableau de
   bord, ou `python scripts/run_radar.py --measure` sans écran.  La carte
   d'état affiche `MEASURING x / 30 s`.  La fenêtre commence avec l'IQ reçu
   après l'appui (le retard de M1 est sauté).
3. **Lire la décision** :
   - `BREATHING` avec la fréquence ;
   - `NO BREATHING` ;
   - `INCONCLUSIVE`, avec la cause :
     - `target_motion` : la cible a bougé ;
     - `saturation` : baisser `sdr.rx_gain_db` ou déplacer le radar ;
     - `too_contaminated` : moins de la moitié de la fenêtre est
       exploitable, quelqu'un a bougé dans le faisceau ;
     - `discontinuity` : des échantillons ont été perdus ;
     - `uncertain` : confiance entre les deux seuils.

     Recommencer la mesure.

   Le résumé de qualité donne la part contaminée, la part saturée et la
   part du temps où quelqu'un bougeait autour du radar.

Le mode continu tourne en parallèle, avec les mêmes traitements.

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
│   │   ├── disturbers.py      marcheurs, balancement, gestes simulés, masquage des trajets
│   │   └── recording.py       écriture et lecture des sessions HDF5 (schéma v1)
│   ├── dsp/                   traitement du signal (ni E/S, ni affichage)
│   │   ├── mixer.py           translation de fréquence (NCO)
│   │   ├── decimation.py      décimateur à état
│   │   ├── doppler.py         M1 passe-bas Doppler, audit du repliement
│   │   ├── contamination.py   M3 indice de contamination, poids, activité
│   │   ├── phase.py           ajustements du cercle, M2 centre glissant, arctangente / DACM, phase → déplacement
│   │   ├── filters.py         passe-bande, suppression de tendance
│   │   ├── estimation.py      périodogramme, Lomb-Scargle pondéré, estimateurs de fréquence, cycles
│   │   ├── detection.py       métriques de respiration et BreathingDetector
│   │   └── spectral.py        fenêtres et TFCT (vue micro-Doppler uniquement)
│   ├── pipeline.py            VitalSignsPipeline : blocs en entrée, PipelineOutput en sortie
│   ├── measurement.py         M4 mode mesure : réglages, résultat, règle de décision
│   ├── ui/
│   │   ├── dashboard.py       tableau de bord de phase temps réel (matplotlib)
│   │   └── launcher.py        écran d'accueil pygame
│   └── ml/                    ancien autoencodeur de spectrogrammes (voir son README)
├── scripts/                   points d'entrée en ligne de commande (voir Démarrage rapide)
├── configs/                   radar.yaml, training.yaml
├── notebooks/                 inference.ipynb (ml/)
├── tests/                     tests pytest, tests/data/references.npz
├── docs/figures/              figures des rapports d'évaluation
├── data/sessions/             sessions HDF5 (ignorées par Git)
├── logs/                      journaux d'exécution (ignorés par Git)
├── pyproject.toml, requirements.txt
├── MOTION_ROBUSTNESS_PLAN.md  conception de la robustesse aux mouvements parasites
└── MOTION_ROBUSTNESS_EVAL.md  son évaluation
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
va dans le journal), `--measure` (une mesure pendant un silence, puis
sortie), `--log-file FICHIER`.  Les journaux vont par défaut dans
`logs/radar_<horodatage>.log`.  Sur le tableau de bord, la touche `m` lance
une mesure.

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

**Évaluer la robustesse aux mouvements parasites** (scènes simulées et
sessions enregistrées, toutes les combinaisons de M1 à M3 ; résultats dans
`results/motion_robustness/`, figures dans `docs/figures/motion_robustness/`) :

```bash
python scripts/evaluate_motion_robustness.py
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
| `motion_robustness` | M1 filtrage Doppler, contrôle du repliement, M2 suivi du centre, M3 pondération (chacun avec `enabled`) |
| `measurement` | M4 : durée (30 s), part valide minimale |
| `micro_doppler_view` | waterfall du tableau de bord (affichage uniquement) |
| `display` | durée d'historique du tableau de bord, plein écran |
| `link_budget` | scénarios optimiste / pessimiste → portée affichée |
| `recording` | dossier des sessions, intervalle d'écriture sur disque |
| `simulation` | scène simulée : présence, chronologie, respiration, cœur, mouvement, fouillis, réflecteurs statiques, perturbateurs, SNR, graine |

Une scène simulée peut évoluer dans le temps, par exemple :

```yaml
simulation:
  enabled: true
  timeline: [[0, breathing], [45, apnea], [60, breathing], [95, motion], [110, empty]]
  disturbers:   # personnes autour du radar (immobiles hors de leur intervalle)
    - {kind: walker, start_m: [-8.0, 3.5], velocity_m_s: [1.0, 0.0], start_s: 30.0, stop_s: 46.0}
    - {kind: gesture, position_m: [0.8, 1.5], start_s: 60.0, stop_s: 90.0}
```

**Principaux paramètres de la robustesse aux mouvements** (`motion_robustness`) :

| Clé | Défaut | Rôle |
|---|---|---|
| `doppler_filter.cutoff_hz` | 1,5 | fréquence à −6 dB de M1 |
| `doppler_filter.stopband_attenuation_db` | 60 | atténuation dès `cutoff + transition/2` = 2 Hz |
| `center_tracking.enabled` | false | M2 (voir plus haut) |
| `center_tracking.window_s`, `step_s` | 10, 1 | sous-fenêtres de M2 (2 à 3 cycles respiratoires) |
| `contamination_weighting.weight_midpoint_db` | 9 | indice de contamination pour lequel le poids vaut 0,5 |
| `contamination_weighting.activity_threshold_db` | 6 | au-dessus, quelqu'un bouge autour du radar |
| `contamination_weighting.activity_lead_s` | 10 | anomalie imputée à un passant vu dans ce délai |
| `contamination_weighting.pause_on_saturation` | true | pause sur saturation de l'ADC |

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

- **carte d'état** : état (et cause d'une pause), fréquence respiratoire,
  jauge de confiance, mode mesure (touche `m` : progression, puis décision),
  bandeau d'apnée, discontinuités du flux ;
- **forme d'onde du déplacement** (brute et filtrée, inspiration vers le
  haut) avec les respirations détectées ;
- **constellation IQ** de la fenêtre analysée avec le cercle ajusté et son
  centre — montre si la phase peut être extraite ;
- **chronologie de la confiance**, colorée par état, avec le poids de M3 ;
- **grandeurs mesurées** : fréquences de quatre estimateurs, intervalle et
  variabilité des respirations, profondeur, rapport I:E, SNR, amplitudes de
  l'écho et du DC, dérive de LO, fréquence cardiaque indicative ; poids
  courant, part contaminée de la fenêtre, saturation ; résumé de la
  dernière mesure ;
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

La suite (environ deux minutes, sans matériel) couvre :

- les briques DSP ;
- les sources, y compris un faux Pluto ;
- les sessions HDF5 et l'équivalence bit à bit entre direct et relecture ;
- la validation de toute la chaîne sur le radar simulé ;
- la robustesse aux mouvements (repliement ≥ 60 dB, suivi du centre,
  marcheurs, gestes, mouvement de la cible, saturation, mode mesure) ;
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
  enregistrements bruts d'une scène contrôlée (`scripts/record.py`).
- **Robustesse aux mouvements : limites d'une seule antenne** (détails dans
  `MOTION_ROBUSTNESS_PLAN.md` §3.4 et §7, figures dans
  `MOTION_ROBUSTNESS_EVAL.md`) :
  - une personne proche du faisceau dont les mouvements restent **dans la
    bande respiratoire** (balancement, report du poids), avec un écho
    comparable à celui de la victime, ne peut pas être supprimée : la mesure
    n'est alors pas concluante.  Seul un filtrage spatial (plusieurs
    antennes, MVDR) pourrait les séparer ;
  - un perturbateur qui **se met à bouger directement dans la bande**, sans
    phase hors bande avant (quelqu'un qui se met à marcher à côté de la
    cible), est pris pour la cible : courte pause ;
  - si la cible bouge **pendant** qu'un passant est actif, l'anomalie est
    imputée au passant : pas de pause, mais le passage reçoit un poids nul ;
  - M1 retarde chaque résultat de 1,82 s et allonge le préchauffage après un
    redémarrage (3,6 s) ;
  - M3 sans M1 laisse les perturbateurs atteindre la démodulation et donne
    de fausses détections dans une pièce vide : garder M1 activé avec M3 ;
  - les seuils (milieu de l'indice, seuil d'activité) sont réglés sur la
    simulation ; ils doivent être vérifiés sur de nouveaux enregistrements
    avec perturbateurs annotés (`/ground_truth/disturbances` n'est écrit que
    pour les sessions simulées).
- **`ml/`** apprend encore sur les anciens spectrogrammes `.npz` de la chaîne
  micro-Doppler supprimée ; il sera repensé sur le signal de phase des
  sessions HDF5.
- **FMCW** (respiration avec sélection en distance) a été étudié mais ne fait
  pas partie de ce paquet ; voir [Points d'attention](#points-dattention).

---

## Licence

Projet interne — usage académique et recherche.
