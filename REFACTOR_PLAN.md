# Plan de restructuration — IoT_radar

> **Version 2** : périmètre validé après vos réponses du 2026-10-01 ; elle remplace la v1.
> **Mise à jour finale** : le §19 fait le bilan (commits, écarts au plan, résultats sur données réelles, points ouverts). Les sections 0 à 18 décrivent le plan ; quand la réalisation s'en écarte, le texte le signale.
> Branche `refactor`, créée depuis `main` = `origin/main` = `8189bbd`.
> Les mesures citées ont été faites en lecture seule, avec des scripts temporaires hors du dépôt.
> Les numéros de ligne renvoient aux fichiers de `8189bbd`.

## Sommaire

0. [Décisions validées](#0-décisions-validées)
1. [État Git et `.claude/`](#1-état-git-et-claude)
2. [Carte des modules actuels et dépendances](#2-carte-des-modules-actuels-et-dépendances)
3. [Redondances](#3-redondances)
4. [Code mort](#4-code-mort)
5. [Bugs et hypothèses discutables](#5-bugs-et-hypothèses-discutables)
6. [`.claude/` : inventaire et décisions](#6-claude--inventaire-et-décisions)
7. [FMCW: findings](#7-fmcw-findings)
8. [Architecture cible](#8-architecture-cible)
9. [Destination de chaque fichier](#9-destination-de-chaque-fichier)
10. [Interface `Source` et blocs](#10-interface-source-et-blocs)
11. [Chaîne de démodulation de phase](#11-chaîne-de-démodulation-de-phase)
12. [Format d'enregistrement HDF5 (schéma v1)](#12-format-denregistrement-hdf5-schéma-v1)
13. [Convention de nommage et renommages](#13-convention-de-nommage-et-renommages)
14. [Dashboard de phase et lanceur](#14-dashboard-de-phase-et-lanceur)
15. [Fonctionnalités retirées](#15-fonctionnalités-retirées)
16. [Tests](#16-tests)
17. [Étapes et commits](#17-étapes-et-commits)
18. [Risques](#18-risques)
19. [Bilan final](#19-bilan-final)

---

## 0. Décisions validées

| # | Décision |
|---|---|
| Q1 | J'ajoute `.claude/` au `.gitignore`. Aucun fichier n'est commité directement depuis `.claude/` : le code utile est **recopié** dans le paquet (puis adapté), et l'original reste en place. Le code mort est supprimé. |
| Q2 | L'équipe passe à la **démodulation de phase**. La chaîne micro-Doppler par spectrogramme sera retirée une fois la chaîne de phase en place et testée. **Le FMCW n'est pas intégré** ; ses conclusions sont consignées au §7. Le dashboard de phase est intégré, sans panneau distance-temps. |
| Q3 | Les correctifs B1 à B5 sont acceptés, un commit par correctif. Pour B2 et B5 seulement, les références de caractérisation peuvent être régénérées dans le même commit (seule exception). |
| Q4 | Les enregistrements passent en **HDF5** (h5py). Le schéma est au §12, les labels au §12.4. Plus aucune écriture de `.npz`, `.iq` ni `.wav`. Les anciens `.npz` restent lisibles par `ml/` ; les anciens `.iq` sont convertibles. |
| Q5 | Tout passe en anglais : code, identifiants, commentaires, docstrings, logs, clés de configuration et **libellés de l'interface**. Convention `snake_case`, unité en suffixe, classes en `CamelCase` (§13). README bilingues. Traduction et renommages font l'objet de commits dédiés, sans changement de logique. |

---

## 1. État Git et `.claude/`

- L'arbre de travail est propre, à part `.claude/` qui n'est pas suivi (`git ls-files .claude` renvoie une liste vide).
- `.claude/worktrees/radar-breath-detection-cleanup-4ec226/` est un **worktree Git enregistré**. Il porte la branche `claude/radar-breath-detection-cleanup-4ec226`, qui pointe sur `8189bbd`. Tout son travail (≈ 4 400 lignes) est **non commité** : supprimer ce dossier le ferait perdre. Je n'y touche pas ; vous le supprimerez vous-même (`git worktree remove --force …`).
- `.claude/` est ajouté au `.gitignore`, et `git add` est toujours utilisé avec des chemins explicites.
- `ed_branch` contient 2 commits absents de `main` (`MicroDopplerDetection/accueil_2.py`, `accueil_tk.py`) : risque de conflit (§18).

---

## 2. Carte des modules actuels et dépendances

Il y a environ 5 100 lignes de Python suivies.

| Fichier | Lignes | Rôle | Importé par |
|---|---:|---|---|
| `MicroDopplerDetection/main.py` | 527 | CLI temps réel et presque tout le reste : YAML (L141), logging (L151), bilan de liaison (L97-134), choix de la source (L215), générateur streaming (L255-410), contexte du dashboard (L417) | `accueil_pg.py` ; `record_acquisition` et `record_visualization`, qui le chargent via `importlib` et appellent ses fonctions **privées** |
| `pipeline/emission.py` | 94 | Buffer TX `cw` ou `cw_offset` (×2¹⁴) | `main`, `legacy` |
| `pipeline/acquisition.py` | 202 | `stream_pluto`, `stream_simulation`, `_check_saturation` | `main` |
| `pipeline/decimation.py` | 190 | `Decimator` à état | `main`, `legacy` |
| `pipeline/clutter.py` | 179 | `ClutterFilter` à état | `main` |
| `pipeline/windowing.py` | 67 | `get_window` | `main`, `legacy` |
| `pipeline/spectrogramme.py` | 87 | `compute_single_column` | `main` |
| `pipeline/detection.py` | 269 | Test F de Fisher, ACF de phase, fusion des scores | `main`, `legacy` |
| `utils/display.py` | 451 | `DashboardRadar` (temps réel et relecture) | `main`, `record_visualization` |
| `utils/record_acquisition.py` | 480 | Enregistrement `.npz`, `.json`, `.iq`, `.wav` | `auto_record` |
| `utils/auto_record.py` | 105 | N enregistrements successifs | — |
| `utils/record_visualization.py` | 264 | Relecture des colonnes d'un `.npz` | — |
| `utils/repo_paths.py` | 14 | Racine des données | les CLI d'enregistrement et de relecture |
| `legacy/*.py` | 585 | Versions « batch » | **aucun appelant** |
| `AICalibration/dataset.py`, `model.py`, `train.py`, `inference.ipynb` | 1 232 | Dataset de spectrogrammes, autoencodeur, entraînement, notebook | — |
| `accueil_pg.py` | 368 | Lanceur pygame | — |

```
accueil_pg ──► main.main (dans un thread : B4)
record_acquisition / record_visualization ──importlib──► main (fonctions privées)
main ──► emission, acquisition ──► decimation ──► clutter ──► windowing + spectrogramme ──► detection ──► display
record_acquisition ══ .npz ══► AICalibration/dataset ──► train ──► results/best.pt ──► notebook
legacy/* ──► pipeline/*        (aucun appelant)
```

---

## 3. Redondances

| # | Constat (vérifié) | Traitement |
|---|---|---|
| R1 | `legacy/*_batch.py` doublonne les briques streaming et ne sert à rien. `decimate_iq` (filtrage à phase nulle) et `compute_spectrogram` (`scipy.signal.stft`, normalisé, bords complétés par des zéros) donnent des résultats **différents** de la chaîne réelle. Les briques streaming traitent déjà un tableau entier en un bloc : `Decimator` donne le même résultat au bit près pour 1, 20 ou 37 blocs ; `ClutterFilter` aussi en modes `butterworth` et `mti`. En modes `iir` et `mean`, l'état initial est la moyenne du premier bloc (`clutter.py` L137, L151-153). | Ajout de `spectral.compute_spectrogram` (hors ligne, identique au flux), puis suppression de `legacy/` |
| R2 | La piste du spectrogramme calculé en double est **infirmée** : l'enregistrement stocke les colonnes, la relecture et le dataset relisent le `.npz`. Seuls `legacy` et l'axe des fréquences (`main` L431, `spectrogramme` L77) sont recalculés. | `spectral.frequency_axis` |
| R3 | Chargement du YAML en 2 versions (`main._load_config` L141, `train._load_yaml` L49) | `config.load_config` |
| R4 | Configuration du logging en 4 versions (`main` L151, `train` L84, `auto_record` L79 avec `basicConfig` ensuite écrasé, `record_visualization` L150) | `config.setup_logging` |
| R5 | 7 manipulations du `sys.path` (`main` L25-28, `record_acquisition` L39-53, `auto_record` L24-35, `record_visualization` L30-46, `train` L34-37, notebook, `PYTHONPATH` exigé par `accueil_pg`) ; `_ensure_paths` copiée ×3 et `_load_main_module` ×2 | `pip install -e .` et `config.REPO_ROOT` |
| R6 | Vitesse de la lumière définie ×5, constantes du CAN ×2 | `physics.py`, `acquisition/pluto.py` |
| R7 | Modes temps réel et relecture de `display.py` : deux encadrés presque identiques (L218-237 et L239-252), une seconde `FuncAnimation` dans `record_visualization` (L248) et l'accès à l'attribut privé `_fig` (L236) | Une seule méthode `run(frames)` |
| R8 | `auto_record` = `record_acquisition` exécuté en boucle | Un seul script, `scripts/record.py -n N` |
| R9 | Le `.json` compagnon est ouvert deux fois (`record_visualization` L79, L97) | Une seule fonction de lecture des métadonnées |

---

## 4. Code mort

Tout le code mort est supprimé (Q1).

| Élément | Décision |
|---|---|
| `legacy/` (5 fichiers, 585 lignes) | Supprimé (étape 1) |
| `accueil_pg.courbe_temps_reel` (L174-206), `etat_ecran` (L85), imports `numpy` et `matplotlib` du lanceur | Supprimés (étape 1) |
| Clé `"signal_iq_dec"` produite par `main` (L402), que personne ne lit | Supprimée (étape 1) |
| Champs `col_complex`, `v_mps`, `df_hz` de `ColumnOutput`, que personne ne lit | Supprimés (étape 1) |
| `Decimator.reset()` (L186) | **Conservé et désormais utilisé** : réinitialisation du pipeline sur une discontinuité (§11.3) |
| `SpectrogramAutoencoder.classify()`, `CalibrationDataset.class_weights` | Supprimés (étape 1) : jamais appelés ; `ml/` sera repensé |
| Valeur `ratio` renvoyée par `_fisher_pvalue` | Disparaît avec Fisher × ACF (étape 6) |

---

## 5. Bugs et hypothèses discutables

| # | Description | Traitement |
|---|---|---|
| **B1** | **Critique, confirmé sur les 69 enregistrements réels.** En `cw_offset`, le buffer TX cyclique contient 4,096 périodes (500 Hz × 16 384 / 2 MS/s). La phase saute de 0,60 rad à chaque répétition, ce qui produit un spectre de raies à k·122,07 Hz : **110,9 dB à 488,28 Hz** contre 39,6 dB à 500 Hz. Le test de Fisher compare deux bandes qui ne contiennent que du bruit ; l'ACF démodule à 500 Hz, la phase dérive à −11,72 Hz, et `acf_peak` vaut 0,867 avec fv = 0,8 Hz pour **tous** les enregistrements testés, labels 0 et 1 confondus. | **Étape 2.** Le décalage est recalé sur un nombre entier de périodes par buffer ; la valeur **effective** est utilisée partout (TX, simulation, réception). `cw_tx_buffer` refuse un décalage non entier. |
| **B2** | Simulation trop optimiste : le fouillis statique est à 0 Hz (`acquisition.py` L179) au lieu de +f_offset, et il n'y a ni DC du récepteur, ni dérive du LO, ni quantification. | **Étape 2.** Simulation physique du worktree, avec en plus la quantification du CAN (§11.4). Références régénérées. |
| **B3** | `np.angle()` est calculé après le passe-haut clutter, ce qui fausse la phase. | **Étape 5, par conception.** La chaîne de phase n'applique aucun passe-haut avant l'extraction de phase (ajustement de cercle). Corriger la branche ACF de la chaîne micro-Doppler modifierait ses références, alors qu'elle est retirée à l'étape 6 ; il n'y aura donc pas de commit séparé, mais le message du commit du pipeline de phase citera B3. |
| **B4** | Le lanceur exécute `main()` dans un thread : Tk tourne hors du thread principal et `argparse` lit la ligne de commande du lanceur. | **Étape 2.** Lancement dans un sous-processus. |
| **B5** | La simulation n'est pas cadencée, alors que l'enregistrement s'arrête sur le temps mur : la durée enregistrée est fausse. | **Étape 2.** Option `realtime` de la simulation. Les valeurs simulées ne changent pas ; les références ne sont régénérées que si un test l'exige. |
| **B6** | Les `.npz` stockent un `config_path` absolu ; la relecture plante dès que le YAML est déplacé. | **Étape 1.** Repli sur la configuration par défaut (non numérique). Avec HDF5, la configuration complète est **stockée dans la session** (attribut `config_yaml`). |
| B7 | Test F : les cases de Hann sont corrélées, donc les p-valeurs sont trop petites. | Disparaît avec Fisher × ACF (étape 6) |
| B8 | Documentation incohérente (λ à 2,4 GHz, `skip_warmup` ≈ 12 au lieu de 24, `hop` 820 au lieu de 819…) | Étape 8 (commentaires seulement) |
| B9 | Débit de `ip:` (Ethernet sur USB) à 2 MS/s | À vérifier sur le matériel |

---

## 6. `.claude/` : inventaire et décisions

Légende des catégories : **(a)** doublon d'une fonctionnalité existante, **(b)** brique générique, **(c)** spécifique au FMCW, **(d)** à ne pas intégrer.
État de `.claude/` : 27 tests sur 27 passent en simulation ; aucune exécution sur le matériel.

| Élément de `.claude/` | Cat. | Décision | Destination |
|---|---|---|---|
| `VitalSigns/phase.py` (ajustement de cercle Kåsa + Gauss-Newton, arctangente, DACM, démodulation linéaire, rotation commune, phase → déplacement) et ses 6 tests | (b) | **Intégré** (étape 5) | `dsp/phase.py` |
| `VitalSigns/estimation.py` (périodogramme, pic FFT, ACF, comptage de pics, passages par zéro, cycles respiratoires) et ses tests | (b) | **Intégré** | `dsp/estimation.py` |
| `VitalSigns/filters.py` : `bandpass`, `widened_band`, `detrend` | (b) | **Intégré** | `dsp/filters.py` |
| `VitalSigns/filters.py` : `StreamingDecimator` | (a) | **Écarté.** `Decimator` est conservé : pour D = 1000 ou 100 000, l'écart entre les formes (b, a) et SOS est inférieur à 5·10⁻¹¹ et le module des pôles reste ≤ 0,966. | — |
| `VitalSigns/detection.py` (SNR, concentration, mouvement ; lissage, hystérésis, apnée) et ses tests | (a) | **Intégré** : **remplace Fisher × ACF**, avec coexistence pendant les étapes 5 et 6 | `dsp/detection.py` |
| `VitalSigns/processor.py` (`VitalSignsProcessor`) et ses tests | (b) | **Base de `pipeline.py`** | `pipeline.py` |
| `VitalSigns/simulation.py` : `target_scene`, chronologie comprise (vide, mouvement, apnée) | (b) | **Intégré** dans la source de simulation CW (étape 5) | `acquisition/sources.py` |
| Simulation physique du worktree (`MicroDopplerDetection/pipeline/acquisition.py` : fouillis à la fréquence TX, DC du récepteur, dérive du LO, `seed`, `realtime`) | (a) | **Intégrée** comme correctif de B2 et B5 (étape 2) | `acquisition/sources.py` |
| `MicroDopplerDetection/pipeline/emission.py` du worktree (`snap_offset`) | (a) | **Intégré** comme correctif de B1 (étape 2) | `acquisition/pluto.py` |
| `MicroDopplerDetection/pipeline/demodulation.py` (`CWDownConverter` = NCO + `StreamingDecimator`) | (a)/(b) | **Réévalué.** L'oscillateur numérique (NCO) est **indispensable** en `cw_offset` : il ramène l'écho de +f_offset à 0 Hz *avant* le filtre passe-bas vers 20 Hz, sinon ce filtre le supprimerait. Le récepteur DC part alors à −f_offset et la décimation le rejette. On reprend le NCO, avec sa phase continue modulo 2π. On ne reprend pas `StreamingDecimator` : la composition NCO + `Decimator` se fait dans le pipeline. | `dsp/mixer.py` (`Mixer`) |
| `MicroDopplerDetection/pipeline/spectrogram.py` (`MicroDopplerSTFT` en temps lent, 256 points, moyenne retirée) | (a)/(b) | **Réévalué.** On garde l'**idée** d'un waterfall micro-Doppler de l'IQ en temps lent (segment de 3,2 s, moyenne retirée pour supprimer la raie de fouillis statique, affichage ±3 Hz), mais calculé avec `dsp/spectral.py` (`compute_single_column`) pour éviter un doublon. Son rôle d'entrée pour l'IA n'est pas repris, car `ml/` sera repensé sur le signal de phase. | `pipeline.py`, via `dsp/spectral.py` |
| `VitalSigns/display.py` (`VitalSignsDashboard`) | (a) | **Intégré** (étape 7), **sans** le panneau distance-temps. Il consomme les dataclasses produites par le pipeline. | `ui/dashboard.py` |
| `accueil_pg.py` du worktree (classe, sous-processus) | (a) | Base du correctif B4 (étape 2) et du lanceur (étape 7), **sans bouton FMCW** | `ui/launcher.py` |
| `VitalSigns/runtime.py`, `paths.py`, `constants.py`, `link_budget.py` | (a) | Versions de référence pour `config.py` et `physics.py` (étape 1) | `config.py`, `physics.py` |
| `VitalSigns/pluto.py` | (a) | Seul le découpage ouverture / flux est repris. La désactivation du suivi DC n'est **pas** reprise (comportement matériel inchangé ; à envisager pour le mode `cw` pur). | `acquisition/pluto.py` |
| `FMCWDetection/*`, `configs/*fmcw*`, `tests/test_chains.py::TestFMCW` | (c) | **Non intégré** (Q2) ; conclusions au §7 | reste dans `.claude/` |
| `MicroDopplerDetection/{main,chain,utils/*,configs}.py` du worktree | (a) | Non repris : remplacés par `pipeline.py`, `scripts/`, `configs/` | — |
| `AICalibration/*` du worktree (modèle 256 × 32) | (d) | Non repris (`ml/` n'est pas redessiné) | — |
| `deletion/`, journaux, `README` | (d) | Non repris (copies identiques ou documentation, reprise à l'étape 8) | — |

---

## 7. FMCW: findings

Conclusions de l'analyse du code FMCW de `.claude/`, gardées pour une future reprise. Ce code reste dans `.claude/` et n'est pas intégré.

1. **Débit USB.** Le dechirping est numérique, faute de mélangeur analogique : il faut donc échantillonner le chirp brut à f_s > B. À 20 MS/s, l'I/Q sur 16 bits représente **80 Mo/s**, bien au-delà de l'USB 2.0 (≈ 35-40 Mo/s utiles). En pratique, le Pluto ne tient en continu que quelques MS/s, et moins encore via `ip:`.
   - Conséquence : acquisition par **rafales**, avec pertes entre les buffers (chaque buffer reste contigu).
   - Le code de `.claude/` réaligne chaque buffer sur le train de chirps, l'horodate avec l'horloge de l'hôte et rééchantillonne le temps lent sur une grille de 20 Hz. **Rien de cela n'a été validé sur le matériel.**
   - L'horodatage côté hôte est pris au retour de `rx()`, pas au moment de l'acquisition (latence de quelques buffers noyau) : la gigue est de l'ordre de la milliseconde, acceptable à 20 Hz.
2. **Bande du chirp.** B = 18 MHz frôle la bande analogique de l'AD9363 (20 MHz) : les extrémités du chirp tombent dans la transition du filtre analogique (atténuation, distorsion de phase), ce qui crée des lobes secondaires en distance. La configuration « large bande » (B = 1 GHz, f_s = 1,2 GS/s, 24 GHz) est purement théorique et impossible sur un Pluto.
3. **Résolution en distance de 8,3 m** (ΔR = c/2B pour B = 18 MHz) : inutilisable pour localiser une victime à quelques mètres. Elle permet au mieux de séparer la fuite TX→RX d'une cible située à plus de 15-20 m.
4. **Masquage par la fuite TX→RX.** Mesure faite avec le code de `.claude/` (fuite d'amplitude 30 à 0,3 m, fenêtre de Hann, FFT sur 512 points) :
   - le profil de la fuite n'est qu'à −2,2 dB à 5 m, −6 dB à 8,3 m, −9,9 dB à 10 m, −24 dB à 15 m, −31,5 dB à 20 m et −43,5 dB à 30 m de son maximum ;
   - une cible **statique** d'amplitude 1 à 15 m ou à 25 m est **masquée** : le maximum dans la porte en distance est trouvé à 5,2 m ;
   - la porte `range_gate_m: [5, 40]` commence **dans** le lobe principal de la fuite ;
   - la case de la cible n'est trouvée que grâce au critère de **variance en temps lent** (cible qui respire, fuite immobile). Sur le matériel, la gigue d'alignement d'un buffer à l'autre rendra la fuite non stationnaire, et la case de fuite risque d'être choisie.
5. **Suivi du DC du récepteur.** Il est désactivé via les attributs IIO `bb_dc_offset_tracking_en` et `rf_dc_offset_tracking_en`, dont le nom dépend du firmware. Non testé.
6. **Pistes pour la suite** : réduire la fuite (antennes TX et RX séparées, ou isolation) ; faire une calibration de fond (soustraire un profil de référence avant la sélection de la case) ; ou opter pour du CW multi-tons (quelques tons, compatible avec le débit USB) pour estimer la distance par différence de phase entre les tons.

---

## 8. Architecture cible

### 8.1 Arborescence (état final, après l'étape 8)

```
IoT_radar/
├── iot_radar/
│   ├── __init__.py
│   ├── config.py           REPO_ROOT, default paths, load_config(), setup_logging()
│   ├── physics.py          speed of light, Boltzmann, T0, wavelength_m(), radar range equation
│   ├── acquisition/
│   │   ├── pluto.py        Pluto configuration, cyclic TX, RX blocks, overflow and saturation checks,
│   │   │                   CW / CW-offset TX buffer, snap_tx_offset_hz() (B1)
│   │   ├── sources.py      Block, Source protocol, PlutoSource, CWSimulationSource (physical model,
│   │   │                   scene timeline, ADC quantization), ReplaySource (HDF5), open_source()
│   │   └── recording.py    HDF5 session writer and reader (schema v1), labels, session naming
│   ├── dsp/
│   │   ├── mixer.py        Mixer: phase-continuous NCO frequency shift
│   │   ├── decimation.py   Decimator (Chebyshev cascade, stateful, reset())
│   │   ├── filters.py      bandpass(), widened_band(), detrend()
│   │   ├── phase.py        circle fit, arctangent, DACM, linear demodulation, LO-drift derotation,
│   │   │                   phase → displacement
│   │   ├── estimation.py   periodogram, rate estimators, breath-by-breath cycles
│   │   ├── detection.py    breathing metrics, BreathingDetector (smoothing, hysteresis, motion, apnea)
│   │   └── spectral.py     windows, STFT column, offline spectrogram, frequency axis (visualization)
│   ├── pipeline.py         VitalSignsPipeline: Block → mixer → decimator → slow-time ring →
│   │                       window analysis → detector → PipelineOutput (dataclasses)
│   ├── ml/
│   │   ├── dataset.py      CalibrationDataset (legacy .npz spectrograms)
│   │   ├── model.py        SpectrogramAutoencoder
│   │   └── train.py        training loop (library)
│   └── ui/
│       ├── dashboard.py    phase dashboard (consumes PipelineOutput, no pipeline import at run time)
│       └── launcher.py     pygame home screen (runs scripts in subprocesses)
├── scripts/                run_radar.py, record.py, replay.py, convert_legacy_iq.py, train.py, launcher.py
├── configs/                radar.yaml, training.yaml
├── notebooks/              inference.ipynb
├── tests/                  pytest, tests/data/ (small golden arrays)
├── data/                   sessions/ (HDF5) and ml/ (legacy .npz), git-ignored
├── results/                training outputs of ml/ (git-ignored)
├── logs/                   run logs (git-ignored)
├── pyproject.toml, requirements.txt
├── README.md, README.fr.md
└── REFACTOR_PLAN.md
```

`dsp/` compte 8 modules. Il n'y aura qu'un pipeline (le CW de phase), donc un seul module `pipeline.py`.

### 8.2 Règles de dépendance

Elles sont vérifiées par un test qui analyse les imports exécutés, y compris transitivement (les imports sous `if TYPE_CHECKING:` sont ignorés).

| Module | N'importe pas |
|---|---|
| `dsp/` | `acquisition`, `ui`, `ml`, `pipeline` |
| `pipeline.py` | `ui`, `ml`. Il n'utilise `acquisition` que pour les annotations de type : les blocs sont lus par attributs. |
| `ui/` | `acquisition` (ni directement, ni transitivement), `ml` |
| `ml/` | `acquisition`, `dsp` |
| `config.py`, `physics.py` | rien du paquet |

C'est `scripts/` qui assemble `acquisition` → `pipeline` → `ui`. **Le pipeline ne connaît pas l'interface** : il renvoie des dataclasses (§11.2), que `ui/dashboard.py` lit.

### 8.3 Écarts à votre proposition initiale

1. `physics.py` est ajouté : constantes et équation radar, qui existaient en 5 copies.
2. `dsp/` gagne `mixer.py`, `filters.py`, `phase.py` et `estimation.py` (chaîne de phase), et perd `clutter.py` à l'étape 6.
3. `configs/`, `notebooks/`, `logs/` et `data/` sont à la racine.
4. `AICalibration/` et `MicroDopplerDetection/` sont **supprimés** en fin de refactor, à votre demande (§19.2). Les sorties d'entraînement vont dans `results/`, les anciens `.npz` sont attendus dans `data/ml/`.
5. `ui/` garde 2 fichiers, comme dans votre proposition.

---

## 9. Destination de chaque fichier

| Actuel | Nouveau | Étape |
|---|---|---|
| `MicroDopplerDetection/main.py` | `config.py` (YAML, logging), `physics.py` (constantes, bilan de liaison), `pipeline.py` (générateur, contexte), `acquisition/sources.py` (`open_source`), `scripts/run_radar.py` (CLI) | 1 |
| `pipeline/emission.py` | `acquisition/pluto.py` | 1 |
| `pipeline/acquisition.py` | `acquisition/pluto.py` (Pluto), `acquisition/sources.py` (simulation) | 1 |
| `pipeline/decimation.py` | `dsp/decimation.py` | 1 |
| `pipeline/clutter.py` | `dsp/clutter.py`, **supprimé** | 1, puis 6 |
| `pipeline/windowing.py` et `spectrogramme.py` | `dsp/spectral.py` | 1 |
| `pipeline/detection.py` | `dsp/detection.py` ; Fisher × ACF remplacé par le nouveau détecteur | 1, puis 5 et 6 |
| `utils/display.py` | `ui/dashboard.py`, réécrit en dashboard de phase | 1, puis 7 |
| `utils/record_acquisition.py` et `auto_record.py` | `acquisition/recording.py` et `scripts/record.py` (HDF5 à partir de l'étape 4) | 1, puis 4 |
| `utils/record_visualization.py` | `scripts/replay.py` (relecture HDF5 via `ReplaySource` à partir de l'étape 4) | 1, puis 4 |
| `utils/repo_paths.py` | `config.py` | 1 |
| `legacy/*` | supprimé | 1 |
| `MicroDopplerDetection/configs/config.yaml` | `configs/radar.yaml` | 1 |
| `MicroDopplerDetection/README.md` et `AICalibration/README.md` | Contenu repris dans `README.md`, `README.fr.md`, `iot_radar/ml/README.md` et `README.fr.md` | 8 |
| `MicroDopplerDetection/logs/.gitkeep` | `logs/.gitkeep` | 1 |
| `AICalibration/dataset.py`, `model.py` | `iot_radar/ml/` | 1 |
| `AICalibration/train.py` | `iot_radar/ml/train.py` et `scripts/train.py` | 1 |
| `AICalibration/config.yaml` | `configs/training.yaml` | 1 |
| `AICalibration/inference.ipynb` | `notebooks/inference.ipynb` | 1 |
| `accueil_pg.py` | `iot_radar/ui/launcher.py` et `scripts/launcher.py` | 1, puis 2 (B4) et 7 |
| `AICalibration/data/`, `AICalibration/results/` | **Supprimés** à votre demande, après conversion des 69 `.iq` dans `data/sessions/legacy/` (§19.2) | fin |

**Correspondance des commandes** (après `pip install -e .`) :

| Avant | Après |
|---|---|
| `python -m MicroDopplerDetection.main --simulation` | `python scripts/run_radar.py --simulation` |
| `python utils/record_acquisition.py --subset train --env salle --label 1 --duration 120` | `python scripts/record.py --label breathing --room salle --duration-s 120` (session HDF5) |
| `python utils/auto_record.py … -n 10 --interval 30` | `python scripts/record.py … -n 10 --interval-s 30` |
| `python utils/record_visualization.py --subset train --index 5` | `python scripts/replay.py data/sessions/session_0005_….h5` |
| `python AICalibration/train.py --epochs 100` | `python scripts/train.py --epochs 100` |
| `python AICalibration/model.py` | `pytest tests/test_ml.py` |
| `python accueil_pg.py` | `python scripts/launcher.py` |
| — | `python scripts/convert_legacy_iq.py <dossier des anciens .iq>` (anciens `.iq` → HDF5) |

---

## 10. Interface `Source` et blocs

```python
@dataclass
class Block:
    """One block of consecutive IQ samples delivered by a source."""
    samples: np.ndarray   # complex64, shape (n_channels, n_samples), in ADC LSB
    sample_start: int     # index of samples[:, 0] in the stream of delivered samples
    host_time_s: float    # host monotonic clock when the block was received (s)
    overflow: bool        # True if samples were lost just before this block


class Source(Protocol):
    sample_rate_hz: float
    n_channels: int
    def read_block(self) -> Block | None: ...   # None = end of data (replay only)
    def describe(self) -> dict[str, Any]: ...   # hardware metadata written into sessions
    def close(self) -> None: ...
```

| Implémentation | Détection d'une discontinuité |
|---|---|
| `PlutoSource` | Avant chaque `rx()`, lecture du registre d'état `0x80000088` du périphérique `cf-ad9361-lpc`, via `sdr._rxadc.reg_read`. Le bit 2 signale une perte d'échantillons ; on l'efface ensuite en réécrivant `0x6`. Si ce registre est inaccessible, un avertissement est émis une seule fois et `overflow` vaut toujours `False`. `sample_start` compte les échantillons **reçus**. **Non testable ici** : à valider sur le matériel. Sur les faibles débits, la fiabilité de ce bit est discutée sur EngineerZone. |
| `CWSimulationSource` | Flux continu ; une perte peut être **injectée** pour les tests (`drop_after_blocks`). |
| `ReplaySource` | Rejoue exactement la table `/blocks` de la session : mêmes frontières de blocs, mêmes `overflow`, mêmes `host_time_s`. |

Le pipeline considère qu'il y a **discontinuité** si `block.overflow` est vrai, ou si `block.sample_start` diffère de l'indice attendu. Il réinitialise alors son état (§11.3).
La forme `(n_channels, n)` est utilisée dès maintenant, avec une seule voie : le passage à 2 voies ne changera pas l'interface. Le pipeline traite la voie 0 ; l'exploitation de plusieurs voies viendra plus tard.

---

## 11. Chaîne de démodulation de phase

### 11.1 Traitement

```
Block (2 MS/s, ADC LSB)
 → Mixer (NCO à −tx_offset_hz : l'écho passe de +f_offset à 0 Hz)       dsp/mixer.py
 → Decimator (D = 2 MS/s / 20 Hz = 100 000 = 2⁵·5⁵)                    dsp/decimation.py
 → anneau du temps lent (fenêtre d'analyse de 20 s ; une sortie toutes les 0,5 s)
 → analyse de la fenêtre (ex-VitalSignsProcessor) :
     dérotation de la dérive du LO → ajustement de cercle (compensation du DC, pas de passe-haut : B3)
     → arctangente / DACM (repli sur la démodulation linéaire si l'arc est trop court)
     → déplacement d = −λφ/4π → passe-bande respiration / cœur
     → spectre du déplacement → métriques (fréquence, SNR, concentration, mouvement)  dsp/phase, filters, estimation, detection
 → BreathingDetector (lissage, hystérésis, MOTION, apnée)                dsp/detection.py
 → colonne micro-Doppler de l'IQ en temps lent (visualisation)            dsp/spectral.py
 → PipelineOutput
```

### 11.2 Dataclasses de résultat

Elles sont définies dans `pipeline.py` ou `dsp/detection.py`, et ne dépendent pas de l'interface.

| Dataclass | Contenu |
|---|---|
| `WindowAnalysis` | `t_s`, déplacement brut et filtré (mm), spectre du déplacement, `BreathMetrics`, `BreathCycles`, IQ analysé, `DemodResult` (ajustement de cercle), fréquences des 4 estimateurs, fréquence cardiaque indicative, dérive du LO |
| `BreathingState` | Sortie du détecteur : état, confiance lissée et instantanée, fréquence (/min), durée de l'épisode, temps depuis la dernière inspiration, apnée |
| `PipelineOutput` | `update_index`, `t_s`, `discontinuity` (un trou a été détecté depuis la sortie précédente), `analysis: WindowAnalysis \| None` (`None` pendant le remplissage), `breathing: BreathingState`, `micro_doppler_column_db` |

### 11.3 Discontinuités

Sur une discontinuité, le pipeline :
- remet à zéro la phase du NCO et l'état du `Decimator` (`reset()`) ;
- vide l'anneau du temps lent et recompte le temps de chauffe ;
- réinitialise le `BreathingDetector` ;
- marque `discontinuity = True` dans la sortie suivante.

L'analyse reprend dès que la fenêtre est de nouveau remplie. Raison : un trou d'une durée inconnue rompt la continuité de phase entre le NCO et l'écho ; une fenêtre qui contiendrait ce saut fausserait l'ajustement de cercle et la phase.

### 11.4 Simulation CW (`CWSimulationSource`)

Modèle physique du worktree, exprimé en LSB du CAN :

```
r[n] = C_rx + (C_static + A · presence(t) · exp(−j4π(R0 + d(t))/λ)) · exp(j2π f_off t) · exp(j2π δf t) + w[n]
```

- `C_rx` : DC du récepteur ;
- `C_static` : fuite et fouillis, **à la fréquence du TX** ;
- `presence(t)` et `d(t)` viennent de `target_scene` : respiration, cœur, chronologie `empty`, `breathing`, `motion`, `apnea` ;
- `δf` : dérive du LO ;
- les échantillons sont ensuite **quantifiés sur les niveaux entiers du CAN** et écrêtés à ±2048. C'est l'ajout de B2 : réalisme, et enregistrement int16 sans perte.

Options : `seed` (`None` par défaut), `realtime` (B5, cadence réelle), injection de pertes pour les tests.
La source expose aussi `ground_truth(t_s)` (déplacement simulé) et `scene_annotations()` (segments étiquetés), écrits dans les sessions simulées.

### 11.5 Sections de `configs/radar.yaml` (état final)

`logging`, `sdr` (`uri`, `center_frequency_hz`, `sample_rate_hz`, `rx_gain_db`, `rx_gain_mode`, `tx_gain_db`, `buffer_size`), `tx` (`waveform`, `offset_hz`), `slow_time` (`rate_hz`, `warmup_s`), `vital_signs` (`window_s`, `update_period_s`, bandes en `_hz`, `dc_compensation`, `demodulation`, `min_arc_rad`, `max_circle_residual`, `lo_drift_compensation`, `bandpass_order`, `detection`), `micro_doppler_view` (`window_s`, `n_fft`, `window`, `display_max_hz`), `display`, `link_budget`, `simulation`, `recording` (`sessions_dir`, `flush_interval_s`).

---

## 12. Format d'enregistrement HDF5 (schéma v1)

### 12.1 Fichier

- **Nom** : `session_<id sur 4 chiffres>_<AAAAMMJJ>_<HHMMSS>.h5`, avec la date et l'heure du début **en UTC**. Exemple : `session_0042_20261005_143012.h5`. `<id>` est le prochain identifiant libre du dossier. La scène ne figure pas dans le nom.
- **Dossier par défaut** : `data/sessions/` (ignoré par Git).
- **Un fichier par session, jamais modifié ensuite** :
  - ouverture en mode `w-`, qui refuse d'écraser un fichier existant ;
  - le fichier est passé en lecture seule (`chmod 0444`) à la fermeture ;
  - les lecteurs ouvrent toujours en mode `r`.
- **Résistance aux coupures** :
  - le format est `libver="latest"` et le mode **SWMR** (single-writer multiple-readers) est activé une fois tous les objets créés ;
  - les données sont ajoutées bloc par bloc, avec un `flush()` toutes les `flush_interval_s` (1 s) ;
  - en cas d'arrêt brutal, le fichier reste lisible jusqu'au dernier `flush()` ;
  - la coupure du lien Pluto et Ctrl-C sont interceptés : le fichier est fermé proprement.

### 12.2 Contenu

| Objet | Type et forme | Contenu |
|---|---|---|
| `/` (attributs) | — | §12.3 |
| `/iq` | `int16`, `(n_channels, n_samples, 2)`, `maxshape=(n_channels, None, 2)`, `chunks=(n_channels, 65536, 2)` | Échantillons I (indice 0) et Q (indice 1). Attributs : `scale` (LSB du CAN par unité stockée : 1,0 pour un enregistrement brut) et `unit = "adc_lsb"` |
| `/blocks` | Table extensible : `sample_start` (`int64`), `host_time_s` (`float64`, secondes depuis le début de la session), `overflow` (`uint8`, 0 ou 1) | Une ligne par bloc reçu : sert à retrouver les discontinuités |
| `/annotations` | Table extensible : `sample_start` (`int64`), `sample_count` (`int64`), `label` (`S16`, ASCII) | Segments étiquetés, écrits à la fin de l'acquisition |
| `/ground_truth/` (optionnel) | `time_s` (`float64`, `(n,)`) et `chest_displacement_m` (`float64`, `(n,)`), échantillonnés à `slow_time.rate_hz` ; attributs `source` (`"simulation"` ou nom du capteur) et `description` | Signal de référence et son horodatage |

Taille : 2 MS/s × 4 octets = **8 Mo/s par voie**, soit 960 Mo pour 2 minutes, sans compression. La compression gzip est écartée pour ne pas surcharger la boucle temps réel. Pour réduire la taille, on pourra baisser `sdr.sample_rate_hz` (par exemple 1 MS/s avec D = 50 000), car la chaîne de phase n'utilise que quelques Hz.

### 12.3 Attributs de la racine

Les valeurs manquantes sont `""` pour un texte et `NaN` pour un nombre.

| Attribut | Type | Origine |
|---|---|---|
| `schema_version` | int | 1 |
| `sample_rate_hz` | float | Débit de `/iq` |
| `center_frequency_hz` | float | LO (TX et RX) |
| `tx_offset_hz` | float | Décalage **effectif** du TX (après correctif B1) ; 0 en `cw` |
| `tx_gain_db`, `rx_gain_db` | float | Configuration |
| `rx_gain_mode` | str | `manual`, `slow_attack`, `fast_attack` ou `hybrid` |
| `n_rx_channels` | int | 1 pour l'instant |
| `channel_layout` | str | Noms des voies dans l'ordre de `/iq`, séparés par des virgules, par exemple `rx0` |
| `firmware_version` | str | Attribut `fw_version` du contexte libiio ; `"simulation"` pour une session simulée |
| `start_time_utc` | str | ISO 8601 |
| `subject_id`, `distance_m`, `orientation`, `obstacle`, `obstacle_thickness_cm`, `room`, `notes` | str / float | Description de la scène (options de la CLI) |
| *Ajouts proposés :* `tx_waveform` | str | `cw` ou `cw_offset` |
| `source_kind` | str | `pluto`, `simulation` ou `legacy_conversion` |
| `iq_stage` | str | `raw_adc`, ou `decimated_clutter_filtered` pour les anciens `.iq` convertis |
| `config_yaml` | str | Texte YAML complet de la configuration utilisée (remplace le chemin absolu : B6) |
| `software_version` | str | Version du paquet et commit Git s'il est disponible |
| `legacy_source_file` | str | Fichier d'origine (conversions seulement) |

### 12.4 Vocabulaire des labels (proposition)

| Label | Signification |
|---|---|
| `empty` | Personne dans le champ du radar |
| `breathing` | Une personne présente et immobile, qui respire |
| `apnea` | Une personne présente et immobile, sans mouvement respiratoire (apnée volontaire) |
| `motion` | Une personne présente, avec des mouvements du corps qui dominent la respiration (mesure impossible) |
| `unknown` | Segment non annoté ou incertain (exclu de l'apprentissage) |

Correspondance avec les anciennes données : label 0 → `empty`, label 1 → `breathing`. On pourrait ajouter plus tard `shallow_breathing` ou `talking` ; je ne les inclus pas sans votre accord. En simulation, les annotations découlent automatiquement de la chronologie de la scène.

### 12.5 Conversion des anciens `.iq` (`scripts/convert_legacy_iq.py`)

Les anciens `.iq` (69 fichiers dans `AICalibration/data/train/`, convertis dans `data/sessions/legacy/` avant la suppression du dossier) contiennent de l'IQ complex64 **décimé à 2 kHz et déjà filtré par le passe-haut clutter**. La conversion :
- lit chaque paire `.iq` + `.json` et écrit une session dans `data/sessions/legacy/` (les fichiers d'origine ne sont pas modifiés) ;
- choisit `scale` de sorte que le maximum de `|I|` et `|Q|` corresponde à 30 000. Le bruit est très au-dessus du pas de quantification, d'environ 30 dB d'après les spectres mesurés ;
- renseigne :
  - `sample_rate_hz` à 2000 ;
  - `iq_stage` à `decimated_clutter_filtered` ;
  - `tx_offset_hz` à 488,28125 Hz : c'est la raie réellement émise à cause de B1, et elle peut être forcée par `--tx-offset-hz` ;
  - les gains et la fréquence porteuse depuis le YAML d'origine s'il existe encore, `NaN` sinon ;
  - `room` depuis `env` ;
  - `start_time_utc` = `utc_finished` − durée ;
  - une seule ligne dans `/blocks` ;
  - une annotation par session (0 → `empty`, 1 → `breathing`).
- **Intérêt** : le passe-haut n'a agi qu'autour de 0 Hz, alors que l'écho est à 488 Hz. Ces enregistrements réels sont donc **retraitables par la chaîne de phase** (D = 100 depuis 2 kHz).

`ml/` continue de lire les anciens `.npz` en l'état.

---

## 13. Convention de nommage et renommages

- **Partout** (fichiers, modules, fonctions, variables, clés de configuration, attributs et datasets HDF5, labels) : anglais, minuscules, mots séparés par `_`, unité en suffixe (`_hz`, `_s`, `_m`, `_mm`, `_db`, `_dbm`, `_dbi`, `_rad`, `_m2`). Les classes sont en `CamelCase`.
- **Paramètres publics** : par exemple `f_s_hz`, `f_c_hz`, `f_s_dec_hz`, `tx_offset_hz`, `wavelength_m`, `breathing_band_hz`, `decimation_factor`.
- **Options de la CLI** : par exemple `--duration-s`, `--interval-s`, `--distance-m`, `--obstacle-thickness-cm`.

Renommage des clés YAML existantes (étape 3) :

| Ancienne clé | Nouvelle clé |
|---|---|
| `sdr.f_c`, `sdr.f_s` | `sdr.center_frequency_hz`, `sdr.sample_rate_hz` |
| `sdr.rx_gain`, `sdr.tx_gain` | `sdr.rx_gain_db`, `sdr.tx_gain_db` |
| `emission.mode`, `emission.f_offset` | `tx.waveform`, `tx.offset_hz` |
| `decimation.enable`, `.D`, `.f_max_utile` | `decimation.enabled`, `.factor`, `.max_useful_frequency_hz` |
| `clutter.butterworth_cutoff` | `clutter.butterworth_cutoff_hz` |
| `windowing.mode` | `spectrogram.window` |
| `spectrogramme.n_fft`, `.overlap`, `.skip_warmup` | `spectrogram.n_fft`, `.overlap`, `.skip_warmup_frames` |
| `detection.bande_respiration`, `.bande_reference` | `detection.breathing_band_hz`, `.reference_band_hz` |
| `detection.alpha`, `.w` | `detection.false_alarm_probability`, `.spectral_weight` |
| `detection.acf_buffer_seconds` | `detection.acf_buffer_s` |
| `affichage.N_historique`, `.seuil_proba`, `.plein_ecran` | `display.score_history_length`, `.score_threshold`, `.full_screen` |
| `bilan_liaison.optimiste`, `.pessimiste`, `.B_eff_hz` | `link_budget.optimistic`, `.pessimistic`, `.noise_bandwidth_hz` |
| `P_tx_dBm`, `G_tx_dBi`, `G_rx_dBi`, `sigma_m2`, `NF_dB`, `L_sys_dB`, `SNR_min_dB` | `tx_power_dbm`, `tx_antenna_gain_dbi`, `rx_antenna_gain_dbi`, `radar_cross_section_m2`, `noise_figure_db`, `system_losses_db`, `min_snr_db` |
| `simulation.enable` | `simulation.enabled` |

Dans `training.yaml`, déjà en anglais, seul `optimizer.lr` devient `optimizer.learning_rate`.
Les sections micro-Doppler (`decimation`, `clutter`, `spectrogram`, `detection`, `display.score_*`) disparaissent à l'étape 6.

---

## 14. Dashboard de phase et lanceur

**Panneaux du dashboard** (repris de `.claude/`, sans le panneau distance-temps) :
1. état (état du détecteur, fréquence par minute, jauge de confiance avec les seuils ON/OFF, durée de l'épisode, bandeau d'apnée) ;
2. forme d'onde du déplacement, brute et filtrée, avec repères de fin d'inspiration et d'expiration ;
3. constellation IQ avec le cercle ajusté et son centre ;
4. chronologie de la confiance et de ses composantes (SNR, concentration), fond coloré selon l'état ;
5. historique de la fréquence respiratoire ;
6. spectre du déplacement (bandes, fenêtre du pic, plancher de bruit) ;
7. waterfall micro-Doppler de l'IQ en temps lent ;
8. grandeurs mesurées ;
9. tableau de paramètres.

Une `discontinuity` est signalée dans le panneau d'état.

Le dashboard est alimenté par un thread producteur qui consomme les `PipelineOutput` et une minuterie qui les dépile, sans perte de trame.

**Lanceur** : boutons « Radar — PlutoSDR », « Radar — simulation », « Dark / light theme » et « Quit » (titre « Modular Radar », sous-titre « S6 project - IoT team »). Le radar est lancé en sous-processus (`scripts/run_radar.py`). Il n'y a pas de bouton FMCW.

---

## 15. Fonctionnalités retirées

| Fonctionnalité | Étape | Remplacement |
|---|---|---|
| Modules batch `legacy/` | 1 | `spectral.compute_spectrogram`, et le traitement d'un tableau entier comme un seul bloc |
| Courbe de démonstration du lanceur | 1 | — |
| Écriture `.npz`, `.iq`, `.wav` ; options `--no-spectrogram`, `--no-iq-file`, `--wav` | 4 | Session HDF5 (IQ brut) |
| Option `--subset train/val/test` | 4 | La répartition se fera à l'entraînement, quand `ml/` sera repensé |
| Options `--env` et `--index` | 4 | `--room` et `--session-id` |
| Chaîne micro-Doppler : décimation à 2 kHz, `ClutterFilter`, spectrogramme de 8192 points, détecteur Fisher × ACF, dashboard TX/RX + score, relecture des colonnes `.npz` dans le dashboard | 6 | Chaîne et dashboard de phase. Les `.npz` restent lisibles par `ml/`. |
| Test de fumée `python model.py` | 1 | `tests/test_ml.py` |
| Relecture d'un `.npz` dans le dashboard (`record_visualization`) | **4** (et non 6) | `scripts/replay.py` sur une session HDF5 ; les anciens `.iq` sont convertibles |
| Dossiers `MicroDopplerDetection/` et `AICalibration/` (README, `results/best.pt`, données `.npz` non suivies) | fin | README bilingues ; `best.pt` reste dans l'historique Git ; `.iq` convertis en HDF5 |

---

## 16. Tests

pytest est installé dans `.venv` ; h5py l'est à l'étape 4.

1. **Caractérisation** (premier commit, avant tout déplacement) : sorties exactes du code actuel sur des entrées déterministes, stockées dans `tests/data/*.npz`. Fonctions couvertes :
   - `Decimator`, `ClutterFilter` (4 modes), fenêtres, colonne STFT ;
   - fonctions de Fisher, d'ACF et de fusion ;
   - buffer TX, bilan de liaison ;
   - générateur streaming de bout en bout, sur une configuration réduite (100 kHz, D = 100) avec une simulation à graine fixée.

   Les paramètres utilisent un `f_offset` **déjà multiple** de f_s/buffer_size : le correctif B1 ne modifie donc pas ces références, et il a son propre test. Seuls B2 et B5 peuvent régénérer les références. Les tests propres au micro-Doppler sont retirés à l'étape 6.
2. **Équivalences** :
   - spectrogramme hors ligne = flux ;
   - `Decimator` en un ou plusieurs blocs ;
   - une session simulée, relue par `ReplaySource` et traitée par le pipeline, donne **exactement** les mêmes sorties qu'en direct.
3. **Structure** : règles de dépendance (§8.2) ; séquence de configuration du Pluto, avec un faux module `adi`.
4. **HDF5** :
   - schéma, attributs, types ;
   - écriture bloc par bloc ;
   - mode `w-` et lecture seule ;
   - relecture après un arrêt simulé ;
   - conversion d'un ancien `.iq` synthétique.
5. **Validation de la chaîne de phase sur simulation** :
   - fréquence respiratoire retrouvée à **±1 cycle/min** ;
   - **amplitude du déplacement** simulé retrouvée (crête à crête, à ±10 %) ;
   - **apnée** signalée ;
   - **absence** → `NO_BREATHING` ;
   - **mouvement** → `MOTION` ;
   - **discontinuité** du flux → réinitialisation, puis nouvelle détection ;
   - tests repris de `.claude/` (`phase`, `estimation`, `detection`, `processor`).
6. **ml** : formes de sortie du modèle et une époque d'entraînement sur de petites données synthétiques.

---

## 17. Étapes et commits

Chaque commit concerne une modification cohérente, avec la suite de tests verte.

**Étape 1 — Restructuration sans changement de comportement** (références intactes)
1. `test: add characterization tests pinning current numerical outputs`
2. `build: add pyproject.toml and iot_radar package; move DSP bricks to iot_radar/dsp/`
3. `feat(dsp): offline spectrogram computed exactly like the stream` puis `refactor: remove dead legacy batch modules`
4. `refactor: move emission, Pluto streaming and simulation to iot_radar/acquisition/`
5. `refactor: split main.py into config, physics, pipeline and scripts/run_radar.py`
6. `refactor: move the dashboard to iot_radar/ui/ and merge live/replay modes`
7. `refactor: merge record_acquisition and auto_record into scripts/record.py; replay in scripts/replay.py`
8. `fix(B6): fall back to the default config when a recording's config path no longer exists`
9. `refactor: move AICalibration code to iot_radar/ml/ and scripts/train.py; single config loader and logging`
10. `refactor: move configs, notebook and launcher; remove dead launcher code and PYTHONPATH manipulations`
11. `refactor: introduce Block and the Source interface (Pluto, simulation)`
12. `test: enforce package dependency rules`

**Étape 2 — Correctifs** :
- `fix(B1): …`
- `fix(B2): … (regenerates simulation references)`
- `fix(B4): …`
- `fix(B5): …`

Pour B3, voir §5 : il est corrigé par conception à l'étape 5.

**Étape 3 — Renommages et traduction** : un commit par sous-paquet (`dsp`, `acquisition`, `pipeline` et `scripts`, `ui`, `ml`), configuration comprise, sans changement de logique.

**Étape 4 — HDF5** :
- `build: add h5py`
- `feat(recording): HDF5 session writer/reader (schema v1)`
- `feat(sources): ReplaySource`
- `refactor(record): sessions instead of .npz/.iq/.wav`
- `feat(replay): replay HDF5 sessions`
- `feat(scripts): convert legacy .iq recordings`

**Étape 5 — Chaîne de phase** :
- un commit par brique `dsp/` (`phase`, `estimation`, `filters`, `mixer`, nouveau détecteur), avec ses tests ;
- `feat(sources): scene timeline and ground truth in the CW simulation` ;
- `feat(pipeline): phase-demodulation pipeline with discontinuity handling (fixes B3 by design)`, avec les tests de validation.

**Étape 6** : `refactor!: remove the micro-Doppler spectrogram chain`, avec le pipeline spectral, Fisher × ACF, `ClutterFilter`, l'ancien dashboard, les tests de caractérisation devenus sans objet et les sections de configuration correspondantes. Entre les étapes 6 et 7, `run_radar.py` et `replay.py` tournent sans interface : ils écrivent l'état dans les logs. L'option `--headless` est conservée ensuite.

**Étape 7** :
- `feat(ui): phase dashboard`
- `feat(ui): launcher without FMCW button`

**Étape 8** :
- docstrings d'en-tête des modules ;
- `README.md` et `README.fr.md` (racine), `iot_radar/ml/README.md` et `README.fr.md` (dépendance aux anciens `.npz` et refonte prévue) ;
- corrections de B8 ;
- résumé final.

Aucune fusion vers `main` sans votre accord.

---

## 18. Risques

| Risque | Mitigation |
|---|---|
| Changement numérique involontaire pendant une étape « sans logique » | Références au bit près, vérifiées à chaque commit |
| Pas de Pluto ici : `PlutoSource` (y compris le registre de perte) n'est pas testée sur le matériel | Séquence de configuration identique, vérifiée avec un faux `adi` ; à valider par l'équipe |
| HDF5 et SWMR : une coupure brutale peut perdre jusqu'à 1 s de données | `flush()` toutes les secondes ; test de relecture après un arrêt |
| Volume des sessions brutes (≈ 1 Go pour 2 min) | Documenté ; possibilité de baisser `sample_rate_hz` |
| `ml/` ne peut pas s'entraîner sur les nouvelles sessions HDF5 | Documenté dans son README ; refonte prévue |
| `AICalibration/results/best.pt` est suivi malgré le `.gitignore` | Supprimé avec le dossier (à votre demande) ; récupérable dans l'historique Git |
| `ed_branch` modifie des chemins qui vont disparaître | À signaler à son auteur |
| `MicroDopplerDetection/logs/*.log` et les `__pycache__/` (non suivis) restent sur le disque | Supprimés avec les dossiers, à votre demande |
| Le passage des libellés de l'interface en anglais peut gêner les utilisateurs francophones | Décision Q5 (« tout le code en anglais ») ; facile à revoir |

---

## 19. Bilan final

### 19.1 Commits (branche `refactor`, non fusionnée)

| Étape | Commits |
|---|---|
| Plan | `4524249`, `af8f556`, `03fb433` (`.claude/` ignoré) |
| 1. Restructuration | `061b5e5` (caractérisation) → `3a91b12` (règles de dépendance), dont `f2e9224` (code mort) et `9e43f08` (axe de fréquence unique), non prévus au §17 |
| 2. Correctifs | `ae90a6e` (B1), `a33b1fb` (B2, références régénérées), `c546e6d` (B4), `3641a0d` (B5, références inchangées) ; B6 en `cfbb61d` |
| 3. Anglais | `a8e026a` (clés YAML) → `37368de` |
| 4. HDF5 | `4ac19b3` → `3c23b16` |
| 5. Chaîne de phase | `c25c562` → `dcac9ef` |
| 6. Retrait micro-Doppler | `571f3b6` |
| 7. Dashboard et lanceur | `7dcccf0`, `27e0c0d` |
| Suppression des anciens dossiers | `065b057` |
| 8. Documentation | `8ba149a` (B8), `7f1c368` (README bilingues), puis la mise à jour de ce plan |

### 19.2 Écarts au plan

1. **`AICalibration/` et `MicroDopplerDetection/` supprimés** (votre message du 2026-10-01 : « les anciennes données .npz ne servent pas vu qu'on passe en HDF5 »). Avant la suppression, les 69 `.iq` ont été convertis dans `data/sessions/legacy/` et comparés aux originaux : pire erreur RMS relative 1,4 × 10⁻⁴. **Attention** : `data/` est ignoré par Git. Ces 69 sessions (70 Mo) n'existent donc que sur cette machine ; à sauvegarder ailleurs si l'équipe veut les garder. Les `.npz` et `.json` d'origine, non suivis, sont supprimés. `best.pt` reste dans l'historique Git.
2. **Relecture des `.npz`** retirée à l'étape 4 (et non 6), en même temps que l'écriture des `.npz` : `scripts/replay.py` ne lit plus que des sessions HDF5.
3. **B6** est rendu sans objet par le HDF5 : la configuration complète est stockée dans la session (`config_yaml`).
4. **Commit `5c2abbc` avec un test rouge** (test du passe-bande trop strict sur le niveau hors bande) : corrigé au commit suivant, `a832b8f`. L'historique n'a pas été réécrit (pas de rebase). Tous les autres commits ont la suite verte.
5. **Validation de l'amplitude** (§16.5) : le test compare l'amplitude crête à crête déduite de l'écart type, et la profondeur des cycles, à ±10 %, à un SNR de −10 dB. Le crête à crête brut est trop sensible au bruit.
6. **Libellés du lanceur** en anglais (§14).
7. `spectral.compute_spectrogram` est conservé : il ne sert plus qu'aux tests et à la visualisation hors ligne d'une session.

### 19.3 Chaîne de phase sur les anciens enregistrements réels

Les 69 sessions converties ont été relues par le pipeline avec la configuration par défaut : fenêtre de 20 s, une analyse toutes les 0,5 s, `tx_offset_hz` = 488,28 Hz. Une session `empty` de 2 s, plus courte qu'une fenêtre, est écartée. Les pourcentages sont moyennés par session.

| Label | Sessions | Analyses | `BREATHING` | `NO_BREATHING` | `MOTION` | Cercle valide | Arc médian | Résidu médian du cercle |
|---|---|---|---|---|---|---|---|---|
| `empty` | 14 | 2 625 | 11,6 % | 70,2 % | 18,2 % | 36 % | 4,43 rad | 0,09 |
| `breathing` | 54 | 10 602 | 1,9 % | 41,1 % | 57,1 % | 63 % | 5,03 rad | 0,24 |

**Lecture.**
- La chaîne **ne détecte pas la respiration** sur ces enregistrements : aucune session `breathing` n'est majoritairement en `BREATHING`.
- Elle se trompe sur 11,6 % des analyses des salles vides.
- Sur les sessions `breathing`, la majorité des fenêtres dépasse le seuil de mouvement (30 mm crête à crête), alors qu'un thorax qui respire bouge de 5 à 15 mm. Deux explications possibles :
  - les sujets bougeaient ;
  - la phase extraite est trop bruitée : le résidu médian du cercle, 0,24, se rapproche des ~0,4 d'un nuage de bruit.

**La cause n'est pas établie.** Ce n'est pas le passe-haut des anciens enregistrements : il n'agit qu'autour de 0 Hz, alors que l'écho est à +488 Hz (§12.5). Ce résultat est négatif. Tant qu'il n'est pas expliqué, la chaîne de phase n'est validée que sur simulation.

### 19.4 Points ouverts

- **Validation matérielle** de `PlutoSource` : séquence de configuration, détection des pertes par le registre `0x80000088`, temps réel à 2 MS/s.
- **Chaîne de phase sur données réelles** : faire de nouveaux enregistrements bruts (`scripts/record.py`) d'une scène connue (personne immobile à 1–2 m, sans obstacle, puis salle vide), et les relire avec `scripts/replay.py`.
- **Refonte de `ml/`** sur le signal de phase des sessions HDF5 (voir `iot_radar/ml/README.md`).
- **`ed_branch`** modifie `MicroDopplerDetection/accueil_2.py` et `accueil_tk.py`, qui n'existent plus : conflit à prévoir.
- **`.claude/`** : à supprimer par vous (le worktree contient du travail non commité, voir §1).
- **Fusion vers `main`** : en attente de votre accord.
