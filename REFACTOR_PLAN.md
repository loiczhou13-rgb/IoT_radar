# Plan de restructuration — IoT_radar (étape 1 : exploration)

> Branche `refactor`, créée depuis `main` = `origin/main` = `8189bbd` (vérifié par `git fetch`).
> À ce stade, **aucun fichier de code n'a été modifié**. Seul ce document a été ajouté.
> Les mesures citées ci-dessous ont été faites en lecture seule, avec des scripts temporaires
> hors du dépôt. Les lignes citées renvoient aux fichiers de `8189bbd`.

## Sommaire

0. [Décisions attendues avant l'étape 2](#0-décisions-attendues-avant-létape-2)
1. [État Git et nature réelle de `.claude/`](#1-état-git-et-nature-réelle-de-claude)
2. [Carte des modules actuels et dépendances](#2-carte-des-modules-actuels-et-dépendances)
3. [Redondances](#3-redondances)
4. [Code mort](#4-code-mort)
5. [Bugs et hypothèses physiques ou matérielles discutables](#5-bugs-et-hypothèses-physiques-ou-matérielles-discutables)
6. [Section `.claude/`](#6-section-claude)
7. [Structure cible](#7-structure-cible)
8. [Destination de chaque fichier](#8-destination-de-chaque-fichier)
9. [Fusions prévues (étape 3)](#9-fusions-prévues-étape-3)
10. [Interface `Source`](#10-interface-source)
11. [Stratégie de test](#11-stratégie-de-test)
12. [Commits prévus](#12-commits-prévus)
13. [Risques](#13-risques)

---

## 0. Décisions attendues avant l'étape 2

| # | Question | Ma recommandation |
|---|----------|-------------------|
| **Q1** | Les consignes se contredisent : « copie ce qui est utile, et supprime l'original en place » (section Git) et « Ne supprime pas `.claude/` : je le ferai moi-même » (étape 4). Or `.claude/` contient un **worktree Git avec ~4 400 lignes de travail non commité** (§1) : le supprimer, c'est le perdre définitivement. | Ne rien supprimer ; je n'y touche qu'en lecture. Si vous le souhaitez, je peux d'abord figer ce travail dans un commit **sur sa propre branche** `claude/radar-breath-detection-cleanup-4ec226` (jamais fusionnée), ce qui vous permettra ensuite de faire `git worktree remove` sans rien perdre. |
| **Q2** | `.claude/` n'est pas qu'un module FMCW. La chaîne FMCW y repose sur **VitalSigns** : démodulation de phase arctangente/DACM, nouveau détecteur et nouveau dashboard. Or c'est précisément l'évolution que vous avez exclue de cette tâche. Que faut-il intégrer à l'étape 4 ? | **(A)** Intégrer le **frontal FMCW** : chirp, simulation, dechirping, FFT en distance et sélection de la case distance. Intégrer aussi les **éléments de visualisation génériques** dans le dashboard existant : waterfall Doppler-temps en CW, carte distance-temps avec marqueur de la case retenue en FMCW, et panneau de paramètres. Le pipeline FMCW produirait alors la carte distance-temps, la distance retenue et le signal temps lent de cette case, **sans détection de respiration** pour l'instant. VitalSigns resterait dans `.claude/` (testé, 27 tests verts) pour la future tâche « démodulation de phase », et la structure cible lui réserve sa place (§7). Autres options : **(B)** tout intégrer dès maintenant, VitalSigns compris ; **(C)** ne rien intégrer du FMCW. |
| **Q3** | Quels bugs corriger ? Chaque correctif serait un commit `fix:` séparé, après l'étape 3, car il modifie des résultats. | Corriger **B1** (critique, confirmé sur les enregistrements réels), **B4** et **B5**. Pour **B2**, corriger seulement si vous voulez une simulation fidèle au matériel ; c'est conseillé, mais cela change le comportement simulé. **B6** est rendu nécessaire par le déplacement des fichiers et n'est pas numérique : je propose de l'inclure dans l'étape 2. |
| **Q4** | La relecture comme `Source` : aujourd'hui, la relecture rejoue des **colonnes de spectrogramme** stockées, pas de l'IQ. Une `ReplaySource` qui produit de l'IQ suppose donc de retraiter le fichier `.iq`. Or ce fichier est enregistré **après** le filtre clutter (`main.py` L350-353) : le pipeline le filtrerait une seconde fois. | Enregistrer désormais l'IQ décimé **avant** le filtre clutter. Cela change le contenu des futurs `.iq` ; un champ `iq_stage` serait ajouté au `.json`. La `ReplaySource` le retraite alors exactement comme en direct, résultats identiques au bit près et score affiché. Sans `.iq`, la relecture des colonnes stockées reste le comportement par défaut. Pour les 69 anciens `.iq` (filtrés), un avertissement signale le double filtrage. Cela prépare aussi la démodulation de phase, qui exige un IQ non filtré passe-haut (B3). |
| **Q5** | Décisions par défaut (dites-moi si l'une ne vous convient pas). | Les données restent dans `AICalibration/data/` et `AICalibration/results/`, sans y toucher. Les YAML vont dans `configs/`, le notebook dans `notebooks/`, les journaux dans `logs/` à la racine. Commentaires, docstrings, messages de log, d'exception et aide CLI passent en anglais ; les **libellés affichés à l'utilisateur final** (dashboard, lanceur) restent en français. J'ajoute `.claude/worktrees/` au `.gitignore`. `ui/` garde 2 fichiers, comme dans votre proposition, ce qui est à la limite de la règle des 3 fichiers. |

---

## 1. État Git et nature réelle de `.claude/`

- `git status` : l'arbre est propre ; seul `.claude/` apparaît comme non suivi, ce qui est la situation que vous aviez annoncée.
- `git ls-files .claude` renvoie une liste vide : **`.claude/` n'est pas suivi** par Git.
- `git worktree list` révèle que `.claude/worktrees/radar-breath-detection-cleanup-4ec226/` est un **worktree enregistré**. Il porte la branche `claude/radar-breath-detection-cleanup-4ec226`, qui pointe sur `8189bbd`, le même commit que `main`. Cette branche n'a donc **aucun commit propre** : tout le travail est dans l'arbre de travail du worktree, soit 20 fichiers modifiés, 13 renommés (dont 11 vers `deletion/`) et une trentaine de fichiers nouveaux non suivis (`FMCWDetection/`, `VitalSigns/`, `tests/`…).
- `main` est à jour avec `origin/main`.
- `ed_branch` contient 2 commits absents de `main`, qui ajoutent `MicroDopplerDetection/accueil_2.py` et `accueil_tk.py`. Ces chemins vont disparaître ; voir les risques (§13).
- Pour éviter tout ajout accidentel du worktree, je n'utiliserai **jamais** `git add -A` : chaque commit indique ses fichiers explicitement.

---

## 2. Carte des modules actuels et dépendances

### 2.1 Fichiers (≈ 5 100 lignes Python suivies)

| Fichier | Lignes | Rôle | Importe (interne) | Importé par |
|---|---:|---|---|---|
| `MicroDopplerDetection/main.py` | 527 | CLI temps réel **et** tout le reste : chargement YAML (L141), logging (L151), bilan de liaison (L97-134), choix de la source (L215), générateur streaming (L255-410), contexte du dashboard (L417) | `pipeline.*`, `utils.display` | `accueil_pg.py` ; `record_acquisition` et `record_visualization`, qui le chargent via `importlib` pour appeler ses fonctions **privées** |
| `pipeline/emission.py` | 94 | Buffer TX `cw` ou `cw_offset`, mis à l'échelle ×2¹⁴ | — | `main`, `legacy/acquisition_batch` |
| `pipeline/acquisition.py` | 202 | `stream_pluto`, `stream_simulation`, `_check_saturation` | — | `main` |
| `pipeline/decimation.py` | 190 | `Decimator` à état (Chebyshev en cascade) | — | `main`, `legacy/decimation_batch` |
| `pipeline/clutter.py` | 179 | `ClutterFilter` à état (`mean`, `iir`, `mti`, `butterworth`) | — | `main` |
| `pipeline/windowing.py` | 67 | `get_window` | — | `main`, `legacy/spectrogramme_batch` |
| `pipeline/spectrogramme.py` | 87 | `compute_single_column` (une colonne STFT) | — | `main` |
| `pipeline/detection.py` | 269 | Test F de Fisher, ACF de phase, fusion des scores | — | `main`, `legacy/detection_batch` |
| `utils/display.py` | 451 | `DashboardRadar` (mode temps réel et mode relecture) | — | `main`, `record_visualization` |
| `utils/record_acquisition.py` | 480 | CLI d'enregistrement `.npz`, `.json`, `.iq` (et `.wav` en option) | `utils.repo_paths`, `main` (importlib) | `auto_record` |
| `utils/auto_record.py` | 105 | CLI qui enchaîne N enregistrements | `record_acquisition` | — |
| `utils/record_visualization.py` | 264 | CLI de relecture d'un `.npz` | `utils.repo_paths`, `utils.display`, `main` (importlib) | — |
| `utils/repo_paths.py` | 14 | `default_recording_data_root()` | — | les deux CLI précédentes |
| `legacy/*.py` (5 fichiers) | 585 | Versions « batch » | `pipeline.*` | **rien** |
| `AICalibration/dataset.py` | 270 | `CalibrationDataset` (fenêtres de colonnes STFT) | — | `train`, notebook |
| `AICalibration/model.py` | 399 | `SpectrogramAutoencoder` et test de fumée | — | `train`, notebook |
| `AICalibration/train.py` | 563 | CLI d'entraînement | `dataset`, `model` | — |
| `AICalibration/inference.ipynb` | — | Courbes, matrice de confusion, reconstructions | `dataset`, `model` | — |
| `accueil_pg.py` | 368 | Lanceur pygame | `MicroDopplerDetection.main.main` | — |

### 2.2 Graphe de dépendances

```
accueil_pg.py ──► main.main  (dans un thread, voir B4)

record_acquisition ─┐  importlib("main.py") : _load_config, _setup_logging,
record_visualization┘                         _build_context, _streaming_frame_generator
auto_record ──► record_acquisition.run

main.py ──► emission ─┐
        ├─► acquisition (Pluto | simulation)
        ├─► decimation ─► clutter ─► windowing + spectrogramme ─► detection
        └─► utils.display

legacy/* ──► pipeline.*   (aucun appelant)

record_acquisition ══ .npz/.json/.iq ══► AICalibration/dataset ──► train ──► results/best.pt ──► notebook
```

### 2.3 Flux de données (CW, configuration actuelle)

Pluto ou simulation (2 MS/s, buffers de 16 384 échantillons) → `Decimator` (D = 1000, soit 2 kHz) → `ClutterFilter` (passe-haut Butterworth à 0,05 Hz) → anneau de `n_fft` = 8192 échantillons avec un pas de 819 → fenêtre de Hann → colonne STFT en dB → détection (test F autour de `f_offset` et ACF de la phase démodulée sur 20 s) → dashboard, ou enregistrement `.npz`.

---

## 3. Redondances

Chaque piste suggérée a été vérifiée ; le verdict figure à chaque fois.

**R1 — `legacy/*_batch.py` et leurs équivalents streaming.** *Confirmé.* Ces fichiers sont du code mort (aucun import hors de `legacy/`, vérifié avec `git grep` dans les `.py` et `.ipynb`).
- `acquisition_batch.acquire_pluto` (L46-131) recopie `acquisition.stream_pluto` (L22-110), y compris le bloc de configuration du Pluto. `_check_saturation` existe deux fois (`acquisition_batch` L185, `acquisition` L190). `synthesize_iq` (L134) applique le même modèle que `stream_simulation` (L113).
- `decimation_batch.decimate_iq` (L24) utilise `scipy.signal.decimate`, à phase nulle : ses résultats **diffèrent numériquement** du `Decimator` causal. `detection_batch` (L38-172) ne contient que des variantes inutilisées. `spectrogramme_batch.compute_spectrogram` (L37) passe par `scipy.signal.stft`, qui normalise par la somme de la fenêtre et complète les bords par des zéros : ses résultats **diffèrent** donc des colonnes streaming.
- Les briques streaming savent **déjà** traiter un tableau complet comme un seul bloc ; je l'ai vérifié :
  - `Decimator` donne le même résultat au bit près pour 1 bloc, 20 blocs égaux ou 37 blocs inégaux ;
  - `ClutterFilter` donne le même résultat au bit près en modes `butterworth` et `mti` ;
  - en modes `iir` et `mean`, l'état initial est la moyenne du **premier bloc** (`clutter.py` L137 et L151-153), d'où un écart de 1,7·10⁻³ sur le transitoire initial. C'est une propriété du filtre, que je documenterai ; ce n'est pas un bug.
- Il manque seulement un spectrogramme hors ligne. Il sera ajouté (F1) en découpant le signal **exactement** comme la boucle streaming, donc avec des colonnes identiques par construction.

**R2 — Calculs de spectrogramme dupliqués entre enregistrement, relecture et dataset.** *Piste en grande partie infirmée.* L'enregistrement stocke les colonnes streaming ; la relecture et le dataset **relisent** le `.npz` sans rien recalculer. Le seul calcul en double est `legacy/spectrogramme_batch` (voir R1). En revanche, l'axe des fréquences est recalculé dans `main._build_context` (L431), à l'identique de `spectrogramme.py` (L77) : une seule fonction sera partagée (F10).

**R3 — Chargement de la configuration.** *Confirmé.* `main._load_config` (L141-148) et `train._load_yaml` (L49-54) font la même chose. `record_acquisition` (L215) et `record_visualization` (L209) appellent la version de `main.py` via `importlib`.

**R4 — Configuration du logging.** *Confirmé, en 4 versions.* `main._setup_logging` (L151-208), `train._setup_logging` (L84-100), `auto_record` avec `basicConfig` (L79-83, ensuite **écrasé** par `_setup_logging` appelé dans `record_run`) et `record_visualization` avec `basicConfig` (L150-154).

**R5 — `repo_paths.py` et manipulations du `PYTHONPATH`.** *Confirmé, en 7 endroits.* On trouve un `sys.path.insert` dans `main.py` (L25-28, avec `_PACKAGE_ROOT` défini deux fois, L25 et L51), `record_acquisition` (L39-53), `auto_record` (L24-35), `record_visualization` (L30-46), `train` (L34-37) et la cellule 1 du notebook. De plus, `accueil_pg.py` exige un `PYTHONPATH` exporté. `_ensure_paths` est copiée **trois fois à l'identique** et `_load_main_module` **deux fois**.

**R6 — Constantes physiques.** *Confirmé.* `_SPEED_OF_LIGHT` est défini 5 fois (`main` L47, `acquisition` L19, `spectrogramme` L16, `legacy/acquisition_batch` L19, `legacy/spectrogramme_batch` L21). `_ADC_FULL_SCALE` et `_ADC_SATURATION_RATIO` le sont 2 fois.

**R7 — Modes « temps réel » et « relecture » de `display.py`.** *Confirmé.*
- La même classe porte un drapeau `show_presence_score`, avec deux listes de texte presque identiques pour l'encadré d'informations (L218-237 et L239-252).
- La relecture pilote sa **propre** `FuncAnimation` (`record_visualization` L248) et accède à l'attribut privé `dashboard._fig` (L236).

**R8 — `record_acquisition.py` et `auto_record.py`.** *Confirmé.* `auto_record` se contente de réutiliser l'analyseur d'arguments de `record_acquisition`, d'y ajouter `-n` et `--interval`, puis d'appeler `run()` en boucle. Une seule commande suffit.

**R9 — Lecture du `.json` compagnon.** `_load_config_path_from_recording` (L79) et `_label_from_json_sidecar` (L97) ouvrent chacune le même fichier.

---

## 4. Code mort

Il s'agit de code jamais importé ni appelé, vérifié avec `git grep` sur les `.py` et `.ipynb`.

| # | Élément | Décision |
|---|---|---|
| D1 | `MicroDopplerDetection/legacy/` (5 fichiers, 585 lignes) | Supprimé à l'étape 3, après l'ajout du spectrogramme hors ligne (F1) |
| D2 | `accueil_pg.courbe_temps_reel` (L174-206) : sinusoïde de démonstration ; la variable `etat_ecran` (L85) ; les imports `numpy` et `matplotlib` qui ne servent qu'à elle | Supprimés |
| D3 | Clés `"signal_iq_dec"` et `"detection"` produites par le générateur (`main` L402, L409), qu'aucun consommateur ne lit | `"detection"` est conservée (diagnostic : alerte de Fisher à α) ; `"signal_iq_dec"` est supprimée |
| D4 | Champs `col_complex`, `v_mps` et `df_hz` de `ColumnOutput`, jamais lus | Conservés : ils sont peu coûteux et documentent la physique (Doppler → vitesse) |
| D5 | `Decimator.reset()` (L186), `SpectrogramAutoencoder.classify()` (L357), `CalibrationDataset.class_weights` (L246) | Conservés : ce sont des API publiques cohérentes, couvertes par des tests |
| D6 | Valeur de retour `ratio` de `_fisher_pvalue`, ignorée par l'appelant | Conservée (diagnostic) |
| D7 | Les 11 fichiers de `.claude/.../deletion/` sont des **copies à l'octet près** de fichiers de `main` | Ignorés |

---

## 5. Bugs et hypothèses physiques ou matérielles discutables

Aucun de ces points n'est corrigé sans votre accord (Q3).

**B1 — CRITIQUE, confirmé sur les données réelles : en mode `cw_offset`, le signal émis n'est pas à 500 Hz.**
- `generate_tx_buffer` (`emission.py` L92-94) remplit le buffer cyclique avec 500 Hz × 16 384 / 2 MS/s = **4,096 périodes**. Il n'y a pas un nombre entier de périodes : à chaque répétition du buffer, la phase saute de 0,60 rad. Le spectre émis devient alors un **spectre de raies** à k·f_s/buffer_size = k·122,07 Hz.
- Moyenne des colonnes de `AICalibration/data/train/1.npz` : **110,9 dB à 488,28 Hz**, 89,5 dB à 366 Hz, 91,7 dB à 610 Hz, et seulement **39,6 dB à 500 Hz**, soit le plancher de bruit.
- Conséquences sur le matériel :
  - le test de Fisher compare deux bandes qui ne contiennent que du bruit (500 ± 0,1-0,8 Hz contre 500 ± 2-5 Hz) ;
  - la branche ACF démodule à 500 Hz au lieu de 488,28 Hz : la phase dérive en rampe à −11,72 Hz. Sur 6 enregistrements réels (labels 0 **et** 1), j'obtiens `acf_peak = 0,867` et `fv = 0,8 Hz` (le bord de bande), **à l'identique pour tous**.

  Le score ACF est donc saturé à 1, le score fusionné vaut au moins 0,5 en permanence, et la décision ne dépend plus que d'un test de Fisher appliqué à du bruit.
- Les 69 enregistrements du dataset ont été acquis avec ce défaut. L'autoencodeur voit le spectre complet (8192 points), raie à 488 Hz comprise.
- Deux correctifs sont possibles :
  - choisir `f_offset` multiple de 122,07 Hz (par exemple 488,28125 Hz) ;
  - recaler automatiquement la fréquence et utiliser la valeur effective en réception (version de `.claude/`).

**B2 — La simulation CW est plus optimiste que le matériel.**
- `stream_simulation` place le fouillis statique à **0 Hz** (`acquisition.py` L179). Physiquement, en `cw_offset`, tout écho (fuite TX→RX, murs) est une copie du signal émis et se trouve donc à +`f_offset`.
- En simulation, le passe-haut à 0,05 Hz élimine parfaitement ce fouillis ; sur le matériel, il ne le fait pas.
- Par ailleurs, la simulation émet un ton parfait à 500 Hz et ne reproduit donc pas B1.

**B3 — Hypothèse discutable : passe-haut avant l'extraction de phase.**
- La branche ACF calcule `np.angle()` **après** le filtre clutter.
- En mode `cw` pur, le passe-haut retire aussi la moyenne du terme utile, ce qui fausse l'angle ; `.claude/tests` contient un test de régression qui le démontre.
- Sans importance pour le spectrogramme, qui n'utilise que le module, mais **bloquant pour la future démodulation de phase**. C'est l'un des motifs de Q4.

**B4 — Le lanceur exécute le radar dans un thread** (`accueil_pg.py` L243-246).
- matplotlib et Tk tournent alors hors du thread principal, ce qui peut geler ou planter la fenêtre.
- `argparse` lit la ligne de commande du lanceur.
- Correctif : lancer un sous-processus (version de `.claude/`). Cela supprime aussi la dépendance de `ui/` vers le pipeline.

**B5 — La simulation n'est pas cadencée.**
- `stream_simulation` produit les données aussi vite que le CPU le permet, alors que `record_acquisition` s'arrête sur le **temps mur**.
- En simulation, un enregistrement « de 120 s » contient donc bien plus de 120 s de signal. `t_wall_s` ne correspond pas au temps du signal, et le `.iq`, tronqué à `durée × f_s_dec` (L364-366), ne correspond pas aux colonnes enregistrées.

**B6 — Le déplacement des fichiers casse la relecture : à traiter à l'étape 2.**
- Les 69 `.npz` stockent un `config_path` **absolu** (`/home/zhoul/IoT_radar/MicroDopplerDetection/configs/config.yaml`), et `record_visualization` quitte en erreur si ce fichier n'existe plus.
- Correctif : se replier sur la configuration par défaut, avec un avertissement. Aucun effet numérique.

**B7 — Hypothèse discutable sur le test F.** Les cases d'une FFT pondérée par une fenêtre de Hann sont corrélées, et les deux bandes latérales sont comptées. Les degrés de liberté `2·n` sont donc surestimés et les p-valeurs trop petites, ce qui favorise les fausses alarmes. Pas de correction prévue ici.

**B8 — Documentation incohérente (corrigée à l'étape 5, commentaires seulement).**
- `config.yaml` : « λ ≈ 12,5 cm à 2,4 GHz » alors que `f_c` vaut 3,5 GHz (λ = 8,57 cm).
- « m = 4πD/λ ≈ 1,0 » : la valeur réelle est 1,47 à 3,5 GHz.
- `skip_warmup` « ≈ 12 trames » : le calcul donne 24.
- « hop = 820 » : la valeur réelle est 819.
- Le README indique « f_c ≈ 2,4 GHz ».
- La docstring de `detect_presence_column` parle de la « même fenêtre temporelle », alors que la phase couvre 20 s et la colonne 4,1 s.

**B9 — À vérifier : débit de `uri: "ip:192.168.2.1"`.** Le lien Ethernet sur USB du Pluto est moins rapide que `usb:` ; à 2 MS/s, la marge est réduite. C'est à vérifier sur le matériel, je ne peux pas le mesurer ici.

---

## 6. Section `.claude/`

### 6.1 Ce que c'est

Une refonte complète, **non commitée**, faite par une session précédente :
- remplacement de la détection par spectrogramme par une **démodulation de phase**, avec un nouveau détecteur et un nouveau dashboard (`VitalSigns/`) ;
- ajout d'une **chaîne FMCW** (`FMCWDetection/`) ;
- réécriture de la chaîne CW (temps lent à 20 Hz, STFT sur 256 points), du lanceur et d'AICalibration (entrée de 256 × 32) ;
- ajout de tests ;
- archivage des fichiers retirés dans `deletion/`.

État vérifié :
- **27 tests sur 27 passent** en simulation (108 s) ;
- **jamais exécuté sur le matériel** : les journaux présents ne viennent que de la simulation ;
- **aucun lien** avec le code de `main`.

### 6.2 Inventaire

Légende des catégories : **(a)** doublon d'une fonctionnalité existante ; **(b)** brique générique réutilisable ; **(c)** spécifique au FMCW et fonctionnel en simulation ; **(d)** à ne pas intégrer.
« Testé » signifie couvert par `.claude/tests`, et « appelé » signifie utilisé par une chaîne de `.claude/`.

#### FMCWDetection/

| Fichier ou fonction | Rôle | Complet | Testé | Appelé | Cat. | Décision |
|---|---|---|---|---|---|---|
| `pipeline/emission.py` : `ChirpParams`, `chirp()` | Paramètres de la rampe (pente, ΔR = c/2B, conversion battement → distance) et chirp analytique à des instants arbitraires | oui | oui | oui | **(b)** | → `dsp/fmcw.py`. C'est des maths pures ; `dsp` en a besoin pour la référence de dechirping. |
| `pipeline/emission.py` : `generate_tx_buffer` | Buffer TX d'un nombre entier de chirps, ×2¹⁴ | oui | indirect | oui | **(c)** | → `acquisition/pluto.py` |
| `pipeline/acquisition.py` : `stream_simulation` | Simulation physique : retards exacts, fuite, réflecteurs statiques, DC du récepteur, graine, cadence temps réel | oui | oui | oui | **(c)** | → `acquisition/sources.py` (`FMCWSimulationSource`) |
| `pipeline/acquisition.py` : `stream_hardware` | Pluto en rafales, horodatées | oui | **non** (pas de matériel) | oui | **(c)** | → `PlutoSource`, commune au CW et au FMCW (`buffer_size` = (chirps + 1)·N) |
| `pipeline/dechirp.py` | `estimate_offset` (alignement sous-échantillon par corrélation circulaire suréchantillonnée), `dechirp`, `lowpass_decimate` (FIR polyphase) | oui | oui (précision < 0,02 échantillon) | oui | **(c)** | → `dsp/fmcw.py` |
| `pipeline/range_processing.py` | `RangeProcessor` (fenêtre, FFT en distance, moyenne cohérente des chirps), `resample_uniform`, `select_range_bin` (variance maximale en temps lent) | oui | oui | oui | **(b)** pour la FFT en distance, **(c)** pour le reste | → `dsp/fmcw.py` |
| `pipeline/chain.py` : `FMCWChain` | Orchestration : profil, historique horodaté, grille à 20 Hz, choix de la case, puis `VitalSignsProcessor` | oui | oui | oui | **(c)** | → `pipeline.py`, **jusqu'à la sélection de la case** ; la suite dépend de Q2 |
| `main.py` | CLI FMCW | oui | non | — | **(c)** | → `scripts/run_fmcw.py` |
| `configs/config.yaml` et `config_wideband_sim.yaml` | Pluto (B = 18 MHz) et démonstration large bande (24 GHz, B = 1 GHz, simulation seulement) | oui | oui | — | **(c)** | → `configs/fmcw.yaml` et `configs/fmcw_wideband_sim.yaml` (section `vital_signs` selon Q2) |
| `logs/*.log`, `README.md` | Journal d'une exécution simulée ; documentation | — | — | — | **(d)** | Le contenu du README est réutilisé à l'étape 5 |

#### VitalSigns/

| Fichier | Rôle | Complet | Testé | Appelé | Cat. | Décision |
|---|---|---|---|---|---|---|
| `constants.py` | c, k, T0, `wavelength()` ; bandes physiologiques | oui | indirect | oui | **(a)** pour les constantes (doublon de R6), **(d)** pour les bandes, inutilisées par `main` | Constantes → `physics.py` |
| `paths.py` | `REPO_ROOT`, racine des données | oui | non | oui | **(a)**, doublon exact de `repo_paths.py` | Fusionné dans `config.py` |
| `runtime.py` | `load_config` (vérifie qu'on obtient bien un dictionnaire) et `setup_logging` (dossier, préfixe ; réduit matplotlib et PIL au niveau WARNING) | oui | non | oui | **(a)**, meilleur que les 4 versions de R3 et R4 | **Base** de `config.py` |
| `link_budget.py` | Équation radar | oui | non | oui | **(a)** : même formule que `main` L97-134, résultats identiques | Structure en module reprise dans `physics.py` ; clés YAML de `main` conservées (`optimiste`/`pessimiste`) |
| `pluto.py` | `open_pluto` et `stream_pluto` séparés, `rf_bandwidth`, désactivation du suivi DC, horodatage, `check_saturation` | oui | **non** (matériel) | oui | **(a)**, mieux découpé que `main` | Découpage repris dans `acquisition/pluto.py`. `disable_dc_tracking` est **désactivé par défaut**, pour ne rien changer sur le matériel sans accord. |
| `filters.py` : `StreamingDecimator` | Même algorithme que `Decimator`, mais en SOS et avec les grands facteurs d'abord | oui | oui | oui | **(a)** | **Non repris.** Résultats différents, et l'argument de stabilité ne tient pas : pour D = 1000, l'écart entre les réponses en forme (b, a) et en SOS est inférieur à 5·10⁻¹¹ et le module des pôles reste ≤ 0,966. `Decimator` est conservé. |
| `filters.py` : `bandpass`, `widened_band`, `detrend` | Filtrage par fenêtre à phase nulle | oui | indirect | oui | **(b)**, mais lié à la phase | Reporté à l'évolution « phase » (Q2) |
| `phase.py` | Ajustement de cercle (Kåsa puis Gauss-Newton), arctangente, DACM, démodulation linéaire, rotation commune, phase → déplacement | oui | oui (6 tests) | oui | **(b)** | **Reporté (Q2)**. Place prévue : `dsp/phase.py` |
| `estimation.py` | Périodogramme, pic FFT, ACF, comptage de pics, passages par zéro, cycles respiratoires | oui | oui | oui | **(b)** | Reporté (Q2) → `dsp/` |
| `detection.py` | Nouveau détecteur (SNR, concentration, mouvement ; hystérésis, apnée) | oui | oui | oui | **(a)**, remplace le Fisher × ACF de `main` | Non intégré : il **change les résultats** et relève de l'évolution « phase » |
| `processor.py` | `VitalSignsProcessor` | oui | oui | oui | **(b)** | Reporté (Q2) |
| `simulation.py` : `target_scene` | Scène simulée : respiration, battements cardiaques, présence, chronologie (vide, mouvement, apnée) | oui | indirect | oui | **(b)** | Partie respiration, cœur et présence → `acquisition/sources.py` pour la simulation FMCW ; la chronologie est reportée |
| `display.py` : `VitalSignsDashboard` | Dashboard en 9 panneaux | oui | non | oui | **(a)**, doublon de `DashboardRadar` | Repris dans `ui/dashboard.py` : panneau **waterfall** (Doppler-temps ou distance-temps, avec marqueur), **tableau de paramètres** (`_table`, `_fmt`) et file de trames dépilée par minuterie, sans perte de trame. Panneaux propres à la phase (constellation IQ, forme d'onde, confiance, rythme) reportés. |

#### Fichiers modifiés de MicroDopplerDetection/ (dans le worktree)

| Fichier | Cat. | Décision |
|---|---|---|
| `pipeline/emission.py` (`snap_offset`) | **(a)**, meilleur | C'est le correctif de **B1**, à appliquer seulement après accord (Q3) |
| `pipeline/acquisition.py` (simulation physique : fouillis à la fréquence TX, DC du récepteur, dérive LO, `seed`, `realtime`) | **(a)**, meilleur | Corrige **B2** et **B5** après accord. Le paramètre `seed` est repris tout de suite, avec la valeur `None` par défaut pour ne rien changer. |
| `pipeline/demodulation.py`, `pipeline/spectrogram.py` | **(d)** pour l'instant | Nouvelle architecture : temps lent à 20 Hz, STFT de 256 points ; change le format d'entrée de l'IA |
| `pipeline/chain.py` (`CWChain`) | **(a)** | Pas repris tel quel. L'idée de `frames_from_slow_time(blocks)`, où le pipeline consomme n'importe quel itérable, inspire l'interface `Source`. |
| `utils/record_*.py`, `utils/auto_record.py`, `configs/config.yaml`, `main.py` | **(a)** | Non repris (liés à la nouvelle chaîne). Idées reprises : relecture cadencée (`_paced`) et retraitement du `.iq`. |

#### Autres fichiers du worktree

| Fichier | Cat. | Décision |
|---|---|---|
| `accueil_pg.py` (classe, sous-processus, boutons CW et FMCW) | **(a)**, meilleur | Base de `ui/launcher.py`. Le passage au sous-processus corrige **B4** (Q3). |
| `AICalibration/*` (traduction anglaise, modèle 256 × 32, suppression de `best.pt`) | **(d)** | Je refais la traduction moi-même. 256 × 32 est lié au nouveau spectrogramme. Je ne touche pas à `results/`. |
| `tests/test_chains.py` | **(c)** pour les tests FMCW, **(d)** pour les tests CW (nouvelle chaîne) | Le test de dechirping (offset et battement → distance) est repris |
| `tests/test_vitalsigns.py` | **(d)** pour l'instant | Repris avec VitalSigns |
| `deletion/`, `.gitignore`, `requirements.txt` | **(d)** | Copies identiques ou modifications triviales |

### 6.3 Hypothèses physiques et matérielles discutables (FMCW)

- **Débit USB.** f_s = 20 MS/s en I/Q sur 16 bits représente 80 Mo/s, bien au-delà de l'USB 2.0 (≈ 35-40 Mo/s utiles). En pratique, le Pluto ne tient en continu que quelques MS/s, et moins encore en `ip:`. Le code l'assume : acquisition en **rafales** avec pertes entre les buffers (contigus individuellement), horodatage côté hôte et rééchantillonnage du temps lent. Tout cela n'a **jamais été validé sur le matériel**.
- **Bande du chirp.** B = 18 MHz atteint presque la limite analogique de l'AD9363 (20 MHz). Les extrémités du chirp tombent dans la transition du filtre analogique, ce qui provoque une atténuation et une distorsion de phase, donc des lobes secondaires en distance.
- **Résolution en distance de 8,3 m** : pour localiser une victime à quelques mètres, c'est inutilisable. Mesure avec le code de `.claude/` :
  - le lobe principal de la fuite TX→RX (amplitude 30, à 0,3 m) n'est qu'à −2 dB à 5 m, −10 dB à 10 m et −24 dB à 15 m ;
  - une cible **statique** à 15 m ou à 25 m est masquée : le maximum est trouvé à 5,2 m ;
  - la porte `range_gate_m: [5, 40]` commence **dans** le lobe principal de la fuite ;
  - la sélection de la case ne fonctionne qu'avec le critère de variance en temps lent (cible qui respire, fuite immobile). Sur le matériel, la gigue d'alignement d'un buffer à l'autre rendra la fuite non stationnaire, et la case de fuite risque d'être choisie.
- **Suivi du DC.** Il est désactivé via les attributs IIO `bb_dc_offset_tracking_en` et `rf_dc_offset_tracking_en`, dont le nom dépend du firmware. Non testé : le code se contente d'un avertissement en cas d'échec.
- **Configuration `config_wideband_sim`** (1,2 GS/s, 24 GHz) : purement théorique, impossible sur un Pluto. Elle est présentée comme telle, ce qui est correct.
- **Horodatage côté hôte.** L'heure enregistrée est celle du retour de `rx()`, pas celle de l'acquisition (latence d'environ 4 buffers noyau). La gigue est de l'ordre de la milliseconde, acceptable pour un temps lent à 20 Hz.

---

## 7. Structure cible

```
IoT_radar/
├── iot_radar/
│   ├── __init__.py
│   ├── config.py          REPO_ROOT, default paths, load_config(), setup_logging()
│   ├── physics.py         c, k, T0, wavelength(), radar range equation (link budget)
│   ├── acquisition/
│   │   ├── pluto.py       open/configure Pluto, cyclic TX + RX blocks, saturation check,
│   │   │                  TX waveforms: CW / CW-offset buffer (+ FMCW chirp buffer, step 4)
│   │   ├── sources.py     Source protocol; PlutoSource, CWSimulationSource, ReplaySource
│   │   │                  (+ FMCWSimulationSource, step 4)
│   │   └── recording.py   .npz/.json/.iq/.wav writers, next index, metadata reader
│   ├── dsp/
│   │   ├── decimation.py  Decimator
│   │   ├── clutter.py     ClutterFilter
│   │   ├── spectral.py    windows, one STFT column, offline spectrogram, frequency axis
│   │   ├── detection.py   Fisher F-test, phase ACF, score fusion
│   │   └── fmcw.py        (step 4) chirp model, dechirp, range FFT, range-bin selection
│   ├── pipeline.py        CW micro-Doppler pipeline (+ FMCW pipeline, step 4) and dashboard context
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py       training loop (library); CLI in scripts/train.py
│   └── ui/
│       ├── dashboard.py   one dashboard for live and replay (+ waterfall panel)
│       └── launcher.py    pygame home screen (launches scripts in subprocesses)
├── scripts/               run_radar.py, record.py, replay.py, train.py, launcher.py (+ run_fmcw.py)
├── configs/               radar.yaml, training.yaml (+ fmcw.yaml, fmcw_wideband_sim.yaml)
├── notebooks/             inference.ipynb
├── tests/                 pytest (+ tests/data/: small golden reference arrays)
├── logs/                  run logs (git-ignored, .gitkeep)
├── AICalibration/         data/ and results/ only — unchanged, not touched
├── pyproject.toml
├── requirements.txt       pinned versions + "-e ."
└── README.md
```

**Règles de dépendance**, qui seront vérifiées par un test analysant les imports :

```
scripts ──► pipeline ──► acquisition ──► dsp ──► physics
   │           │                           ▲
   │           └──────────────────────────┘
   ├──► ui        (ui n'importe ni acquisition, ni pipeline, ni ml)
   ├──► ml        (ml n'importe ni acquisition, ni dsp)
   └──► config    (importable partout ; n'importe rien du paquet)
dsp n'importe ni acquisition, ni ui, ni ml.
```

**Écarts à votre proposition, et pourquoi :**
1. **`physics.py`** est ajouté : il évite 5 copies de `c` (R6) et donne un module propre à l'équation radar, au lieu de l'enfouir dans `pipeline.py`. `dsp/spectral.py` en a besoin pour convertir Doppler → vitesse, ce qui exclut de le placer dans `acquisition/`.
2. **Le modèle mathématique du chirp va dans `dsp/fmcw.py`** et non dans `acquisition/pluto.py` : le dechirping a besoin du chirp de référence, et `dsp` ne doit pas importer `acquisition`. `pluto.py` ne fait que mettre à l'échelle du CNA et émettre.
3. **`configs/`, `notebooks/` et `logs/`** sont à la racine : ce sont des fichiers de données ou de sortie, pas du code du paquet.
4. **`AICalibration/`** ne garde que `data/` et `results/` pour ne pas toucher aux données ; le chemin est configurable dans `training.yaml`.
5. **`ui/`** ne compte que 2 fichiers. C'est votre proposition ; l'autre option serait d'aplatir en `iot_radar/dashboard.py` et `iot_radar/launcher.py`.
6. **`pipeline.py`** contiendra 2 pipelines (CW et FMCW). Selon votre règle, cela ne justifie pas encore un dossier `pipelines/`.
7. **Pas de `[project.scripts]`.** Les commandes sont des fichiers `scripts/*.py` explicites, faciles à lire et à lancer.

**Place prévue pour les évolutions futures, sans remettre la structure en cause :**
- démodulation de phase : `dsp/phase.py` et `dsp/vital_signs.py`, avec un back-end « phase » dans `pipeline.py` ;
- deux voies de réception : `read_block()` renverra `(n_voies, n)` ;
- CW multi-tons : `acquisition/pluto.py` (forme d'onde) et `dsp/` (séparation des tons) ;
- FMCW : déjà prévu.

---

## 8. Destination de chaque fichier

| Actuel | Nouveau |
|---|---|
| `MicroDopplerDetection/main.py` | Découpé ainsi : `_load_config` et `_setup_logging` → `config.py` ; `_radar_range`, `_compute_range` et les constantes → `physics.py` ; `_resolve_f_offset`, `_auto_skip_warmup`, `_streaming_frame_generator` et `_build_context` → `pipeline.py` ; `_build_iq_stream` → `acquisition/sources.py` (`open_source`) ; `_parse_args` et `main` → `scripts/run_radar.py` |
| `pipeline/emission.py` | `acquisition/pluto.py` (`cw_tx_buffer`) |
| `pipeline/acquisition.py` | `stream_pluto` et `_check_saturation` → `acquisition/pluto.py` ; `stream_simulation` → `acquisition/sources.py` |
| `pipeline/decimation.py` | `dsp/decimation.py` |
| `pipeline/clutter.py` | `dsp/clutter.py` |
| `pipeline/windowing.py` et `pipeline/spectrogramme.py` | `dsp/spectral.py` |
| `pipeline/detection.py` | `dsp/detection.py` |
| `utils/display.py` | `ui/dashboard.py` |
| `utils/record_acquisition.py` et `utils/auto_record.py` | Fonctions d'écriture → `acquisition/recording.py` ; CLI → `scripts/record.py` |
| `utils/record_visualization.py` | Lecture des métadonnées → `acquisition/recording.py` ; CLI → `scripts/replay.py` |
| `utils/repo_paths.py` | `config.py` |
| `legacy/*` | Supprimé (D1) |
| `MicroDopplerDetection/configs/config.yaml` | `configs/radar.yaml` |
| `MicroDopplerDetection/README.md` | `iot_radar/README.md` (chaîne de traitement et physique) |
| `MicroDopplerDetection/logs/.gitkeep` | `logs/.gitkeep` |
| `AICalibration/dataset.py`, `model.py` | `iot_radar/ml/dataset.py`, `iot_radar/ml/model.py` (le test de fumée devient un test pytest) |
| `AICalibration/train.py` | Boucle d'entraînement → `iot_radar/ml/train.py` ; CLI → `scripts/train.py` |
| `AICalibration/config.yaml` | `configs/training.yaml` (les chemins restent relatifs à la racine du dépôt) |
| `AICalibration/inference.ipynb` | `notebooks/inference.ipynb` (imports `iot_radar.ml.*`, plus de `sys.path`) |
| `AICalibration/README.md` | `iot_radar/ml/README.md` |
| `AICalibration/data/`, `AICalibration/results/` | **Inchangés** |
| `accueil_pg.py` | `iot_radar/ui/launcher.py` et `scripts/launcher.py` |
| `README.md`, `requirements.txt`, `.gitignore` | Mis à jour ; ajout de `pyproject.toml` |

**Correspondance des commandes :**

| Avant | Après (après `pip install -e .`, depuis la racine) |
|---|---|
| `export PYTHONPATH=$(pwd)`, puis `python -m MicroDopplerDetection.main --simulation` | `python scripts/run_radar.py --simulation` (options `--config` et `--log-file` inchangées) |
| `python utils/record_acquisition.py --subset train --env salle --label 1 --duration 120` | `python scripts/record.py --subset train --env salle --label 1 --duration 120` |
| `python utils/auto_record.py --subset train -n 10 --interval 30 …` | `python scripts/record.py --subset train -n 10 --interval 30 …` |
| `python utils/record_visualization.py --subset train --index 5` | `python scripts/replay.py --subset train --index 5` |
| `python AICalibration/train.py --epochs 100` | `python scripts/train.py --epochs 100` |
| `python AICalibration/model.py` (test de fumée) | `pytest tests/test_ml.py` |
| `python accueil_pg.py` | `python scripts/launcher.py` |

---

## 9. Fusions prévues (étape 3)

Chaque fusion fait l'objet d'un commit séparé, testé. Les fusions écartées figurent à la fin de la section.

| # | Ce qui est fusionné | Pourquoi la version fusionnée est plus simple |
|---|---|---|
| F1 | Ajout de `spectral.compute_spectrogram(iq, …)`, un spectrogramme hors ligne qui découpe le signal comme la boucle streaming ; puis suppression de `legacy/` | Une seule définition du spectrogramme. Un tableau entier se traite en un seul bloc ; un test vérifie que le résultat est identique au flux. |
| F2 | `main._load_config` et `train._load_yaml` → `config.load_config` | Une seule fonction, qui vérifie aussi que le YAML est bien un dictionnaire ; message d'erreur conservé |
| F3 | Les 4 configurations du logging → `config.setup_logging(level, log_file, console_level, file_mode)` | Un seul format. L'entraînement garde une console silencieuse (WARNING) et un fichier `train.log` en ajout. |
| F4 | `repo_paths`, les 7 manipulations de `sys.path`, `_ensure_paths` (×3) et `_load_main_module` (×2) → `pip install -e .` et `config.REPO_ROOT` | Supprime environ 80 lignes de code fragile et les accès aux fonctions privées de `main.py` |
| F5 | `_build_iq_stream` et les deux générateurs (Pluto, simulation) → interface `Source` (§10) ; `pipeline.frames(source)` | Le pipeline ignore d'où viennent les données |
| F6 | Modes temps réel et relecture de `DashboardRadar` → une seule méthode `run(frames, frame_interval_s=None)`, un titre public, un seul constructeur de l'encadré d'informations | Supprime la `FuncAnimation` en double et l'accès à `_fig` |
| F7 | `record_acquisition` et `auto_record` → `scripts/record.py` (`-n`, 1 par défaut ; `--interval` ; `--index` refusé si n > 1) | Une seule commande |
| F8 | Lecture du `.json` compagnon (R9) → `recording.load_recording_metadata()` | Le fichier n'est plus ouvert deux fois |
| F9 | `_SPEED_OF_LIGHT` (×5), les constantes du CAN (×2) et l'équation radar → `physics.py` et `pluto.py` | Une seule définition de chaque grandeur |
| F10 | Axe des fréquences (`_build_context` L431 et `spectrogramme` L77) → `spectral.frequency_axis(n_fft, f_s)` | Une seule définition |

**Fusions écartées :**
- **Spectre TX** (`20·log10(|X|+ε)`) et **colonne RX** (`10·log10(|X|²+ε)`) : les deux formules diffèrent près du plancher ; les fusionner modifierait les valeurs.
- **`Decimator`** (IIR causal, du CAN au temps lent) et **`lowpass_decimate` du FMCW** (FIR sur le temps rapide d'un chirp) : physiquement différents.
- **Retrait de la moyenne dans `select_range_bin`** et **`ClutterFilter`** : rôles différents.

---

## 10. Interface `Source`

```python
class Source(Protocol):
    """Anything that delivers consecutive blocks of complex baseband IQ samples."""
    sample_rate_hz: float          # rate of the samples returned by read_block()
    last_block_time_s: float       # time of the first sample of the last block (s)

    def read_block(self) -> np.ndarray:
        """Next 1-D complex64 block; an empty array means 'end of data'."""

    def close(self) -> None:
        """Release the hardware or the file."""
```

| Implémentation | `sample_rate_hz` | Fin des données | Remarque |
|---|---|---|---|
| `PlutoSource` | `sdr.f_s` | jamais | Configuration du Pluto **identique** (même séquence d'attributs, vérifiée par un test avec un faux module `adi`) ; `last_block_time_s` = horloge de l'hôte |
| `CWSimulationSource` | `sdr.f_s` | jamais | Modèle actuel inchangé ; `seed` optionnel (`None` par défaut, comme aujourd'hui) |
| `ReplaySource` | `f_s_dec_hz` lu dans le `.json` | fin du `.iq` | Le pipeline calcule D = `sample_rate_hz / f_s_dec`, soit D = 1 en relecture. Le `Decimator` vaut alors l'identité. |

Le pipeline ne connaît que `Protocol` : il n'y a pas d'héritage. Pour l'extension à deux voies de réception, `read_block()` renverra `(n_voies, n)`.

---

## 11. Stratégie de test

pytest sera installé dans `.venv`, qui ne l'a pas encore.

1. **Avant tout déplacement**, un premier commit ajoutera des **tests de caractérisation**. Les sorties du code actuel, sur des entrées déterministes, seront stockées dans `tests/data/*.npz` (de petite taille : configuration de test réduite). Les fonctions couvertes sont :
   - `Decimator`, `ClutterFilter` (4 modes), `get_window`, `compute_single_column` ;
   - `_fisher_pvalue`, `_acf_peak`, `_fusion_score` ;
   - le générateur streaming complet sur une simulation à graine fixée ;
   - le buffer TX et le bilan de liaison ;
   - le découpage en fenêtres de `CalibrationDataset` sur des `.npz` synthétiques ;
   - les formes de sortie du modèle.

   Chaque commit suivant doit reproduire ces sorties **au bit près**. C'est la preuve qu'aucun résultat numérique n'a changé.
2. **Tests d'équivalence** :
   - le spectrogramme hors ligne est identique au flux ;
   - le `Decimator` donne le même résultat en un ou plusieurs blocs ;
   - un enregistrement simulé court, relu via `ReplaySource`, reproduit les colonnes stockées (si Q4 = recommandation).
3. **Tests de structure** : les règles de dépendance de §7, par analyse des imports, et la configuration du Pluto avec un faux module `adi`.
4. **Étape 4 (FMCW)**, sur données simulées :
   - paramètres du chirp (ΔR, pente) ;
   - estimation du décalage à mieux que 0,05 échantillon ;
   - conversion battement → distance ;
   - **cible simulée à distance connue retrouvée à la bonne distance**, à ±ΔR/4, en configuration large bande et en configuration Pluto (cible respirante, sélection par variance) ;
   - case de fuite exclue par la porte en distance.
5. **Durée** : la suite doit rester sous la minute environ ; les simulations utilisent `realtime = False` et des configurations réduites.

---

## 12. Commits prévus

**Étape 2 : déplacements sans changement de logique, tests verts à chaque commit**
1. `test: add characterization tests pinning current numerical outputs`
2. `build: add pyproject.toml for editable install (pip install -e .)`
3. `refactor: move DSP bricks to iot_radar/dsp/`
4. `refactor: move emission, Pluto streaming and simulation to iot_radar/acquisition/`
5. `refactor: split main.py into config, physics, pipeline and scripts/run_radar.py`
6. `refactor: move dashboard and launcher to iot_radar/ui/`
7. `refactor: move recording helpers to acquisition/recording.py and CLIs to scripts/`
8. `fix: fall back to the default config when a recording's config path no longer exists` (B6)
9. `refactor: move AICalibration code to iot_radar/ml/ and scripts/train.py`
10. `refactor: move configs and notebook; drop PYTHONPATH manipulations`

**Étape 3** : F1 à F10, un commit par fusion. **Puis les correctifs validés en Q3**, un commit `fix:` chacun.

**Étape 4** : un commit par élément intégré (modèle du chirp, dechirping, FFT en distance et sélection, sources FMCW, pipeline FMCW, waterfall du dashboard, script `run_fmcw`), chacun avec son test.

**Étape 5** :
- traduction en anglais des commentaires et messages restants, un commit par sous-paquet ;
- docstrings d'en-tête de chaque module ;
- README de la racine, de `iot_radar/` et de `iot_radar/ml/` ;
- corrections de B8.

---

## 13. Risques

| Risque | Mitigation |
|---|---|
| Changement involontaire d'un résultat numérique pendant un déplacement | Tests de référence au bit près (§11.1) exécutés à chaque commit |
| Relecture des 69 enregistrements existants (chemin absolu de la configuration) | B6 traité à l'étape 2 ; les clés du `.npz` restent inchangées ; `CalibrationDataset` lit les mêmes clés |
| Pas de Pluto disponible ici, donc la `PlutoSource` refactorisée n'est pas testée sur le matériel | Séquence de configuration conservée à l'identique et vérifiée avec un faux `adi` ; à valider par l'équipe sur le matériel |
| `AICalibration/results/best.pt` (12 Mo) est suivi par Git malgré le `.gitignore` | Non touché. J'ai vérifié qu'il ne référence aucun module Python (seulement `torch` et `collections`) : le notebook pourra toujours le charger après déplacement. |
| `ed_branch` modifie des fichiers sous `MicroDopplerDetection/` | Conflits possibles si elle est fusionnée plus tard ; à signaler à son auteur |
| Ajout accidentel du worktree `.claude/` dans un commit | `git add` toujours avec des chemins explicites ; `.claude/worktrees/` ajouté au `.gitignore` (Q5) |
| `MicroDopplerDetection/logs/*.log` et `__pycache__/` (non suivis) resteront sur le disque après le déplacement | Je ne supprime pas vos fichiers non suivis ; à effacer à la main si vous le souhaitez |
| `pip install -e .` nécessite setuptools (présent dans `.venv`) | En cas d'absence de réseau : `pip install -e . --no-build-isolation` |
| Notebook : `*.ipynb` est ignoré par Git, mais `inference.ipynb` est suivi | Déplacé avec `git mv`, il reste suivi |
