# Radar Micro-Doppler — Détection de survivants ensevelis

Système radar portable à onde continue (CW) pour détecter la respiration de
personnes ensevelies sous des décombres (séisme, avalanche, effondrement).

Matériel visé : **PlutoSDR** (ADALM-PLUTO, AD9363).  
Orchestration **streaming** dans **`main.py`**.

---

## Contexte physique

Un signal CW à **f_c ≈ 2.4 GHz** (λ ≈ 12.5 cm) illumine la scène. Le mouvement
thoracique (**f_v ≈ 0,2–0,5 Hz**, amplitude **D ≈ 5–15 mm**) module la phase du
signal reçu. En bande de base :

```
x(t) = exp(j·φ(t))     avec  φ(t) = φ₀ − (4πD/λ)·sin(2π·f_v·t)
```

Les raies micro-Doppler autour de **±f_v** sont analysées après STFT.

---

## Architecture du pipeline (streaming)

La fenêtre FFT utilisée pour la colonne spectrale est choisie dans
**`configs/config.yaml`** (`windowing.mode`) et instanciée par **`pipeline/windowing.py`** (`get_window`), puis passée à **`spectrogramme.compute_single_column`** — ce n’est pas une étape “disque” séparée, mais un bloc fonctionnel distinct entre le segment IQ et la FFT.

```
┌─────────────┐      ┌────────────────┐      ┌──────────────┐
│ emission.py │─────▶│ acquisition.py │─────▶│decimation.py │
│  buffer TX  │      │ IQ Pluto / sim │      │ Decimator    │
│  ×2¹⁴ scale │      │                │      │ stateful     │
└─────────────┘      └────────────────┘      └──────┬───────┘
                                                    │
                                            ┌───────▼────────┐
                                            │   clutter.py   │
                                            │ ClutterFilter  │
                                            └───────┬────────┘
                                                    │ buffer n_fft + hop
                      ┌─────────────────────────────┼─────────────────────────┐
                      │                             │                         │
                      │                      ┌──────▼───────┐                  │
                      │                      │ windowing.py │ ← config YAML    │
                      │                      │  get_window  │                  │
                      │                      └──────┬───────┘                  │
                      │                             │ fenêtre × segment IQ    │
                      │                      ┌──────▼───────┐                  │
                      │                      │spectrogramme │                  │
                      │                      │compute_single│                  │
                      │                      │   _column    │ → col_db         │
                      │                      └──────┬───────┘                  │
                      └─────────────────────────────┼─────────────────────────┘
                                                    │
                                       ┌────────────▼─────────────┐
                                       │      detection.py       │
                                       │ Fisher F-test + ACF     │
                                       │ → score (temps réel     │
                                       │    uniquement, pas dans │
                                       │    les .npz prod.)       │
                                       └────────────┬─────────────┘
                                                    │
                                       ┌────────────▼─────────────┐
                                       │    utils/display.py    │
                                       │ Dashboard matplotlib   │
                                       └────────────────────────┘
```

**Ordre logique dans `main.py`** : après décimation et clutter, un **anneau** de
longueur `n_fft` avance par pas `hop`. Sur chaque fenêtre valide (hors warm-up)
: **segment** → multiplication par la **fenêtre** → **STFT** (une colonne) → **détection**.

---

## Arborescence utile

```
MicroDopplerDetection/
├── main.py                 # CLI streaming — point d’entrée principal
├── accueil_pg.py           # interface utilisateur (hors chaîne temps réel)
├── configs/
│   └── config.yaml        # source de vérité pipeline + fenêtrage + STFT + détection
├── logs/                  # radar_<timestamp>.log si activé dans la config
├── pipeline/
│   ├── emission.py
│   ├── acquisition.py
│   ├── decimation.py
│   ├── clutter.py
│   ├── windowing.py       # Hann / Hamming / rectangular …
│   ├── spectrogramme.py   # colonne STFT (consomme la fenêtre)
│   └── detection.py
├── utils/
│   ├── display.py         # Dashboard temps réel ou relecture sans panneau score
│   ├── record_acquisition.py
│   ├── record_visualization.py  # relecture .npz (+ label dans le titre si présent)
│   ├── auto_record.py     # plusieurs prises d’affilée (--samples, --interval)
│   ├── repo_paths.py      # défaut AICalibration/data
│   └── migrate_data_root_in_recordings.py  # migration chemins (usage ponctuel)
└── legacy/                # batch hors-ligne — non utilisé par main.py
```

Les enregistrements **supervisés** (`record_acquisition`) sont par défaut sous le
dépôt voisin **`../AICalibration/data/<train|test|val>/<index>.*`**. Voir
**`utils/repo_paths.default_recording_data_root()`**.

---

## Installation

### Prérequis système

```bash
sudo apt install libiio-dev libiio-utils
iio_info -s
```

### Dépendances Python

```bash
pip install numpy scipy matplotlib pyyaml pyadi-iio
```

| Paquet       | Rôle                                              |
|--------------|---------------------------------------------------|
| `numpy`      | IQ, tableaux                                       |
| `scipy`      | décimation IIR, fenêtres via spectrogramme, STFT   |
| `matplotlib` | dashboard                                          |
| `pyyaml`     | configuration                                      |
| `pyadi-iio`  | Pluto                                              |

---

## Utilisation

Depuis la racine du dépôt `IoT_radar` :

```bash
export PYTHONPATH="$(pwd)"
python -m MicroDopplerDetection.main
```

Depuis ce dossier :

```bash
cd MicroDopplerDetection
python main.py
```

### Mode simulation

```bash
python main.py --simulation
```

### Configuration

```bash
python main.py --config configs/mon_setup.yaml
```

`--simulation` force le mode simulé quel que soit le YAML.  
Journal par défaut : `logs/radar_<timestamp>.log` — voir `logging` dans `config.yaml` et `--log-file`.

### Options CLI (`main.py`)

| Option         | Défaut                         | Description               |
|----------------|-------------------------------|---------------------------|
| `--config`     | `configs/config.yaml`         | Fichier YAML              |
| `--simulation` | *(absent)*                    | Forcer la simulation      |
| `--log-file`   | horodaté dans `logs/`         | Fichier de log explicite |

---

## Enregistrement & relecture (hors GUI)

| Script | Rôle |
|--------|------|
| **`utils/record_acquisition.py`** | Une prise : spectrogramme + métadonnées + IQ optionnel sous `AICalibration/data/...` |
| **`utils/auto_record.py`** | Plusieurs prises (`-n`, `--interval`) — mêmes options que ci-dessus |
| **`utils/record_visualization.py`** | Replay d’un `.npz` avec dashboard (sans courbe de score) ; affiche le **label** dans le titre |

Exemples :

```bash
cd MicroDopplerDetection
python utils/record_acquisition.py --subset train --env salle --label 1 --duration 120
python utils/record_visualization.py --subset train --index 5
```

---

## Paramètres (`configs/config.yaml`)

| Section         | Rôle |
|-----------------|------|
| `sdr`           | `f_c`, `f_s`, gains, `uri`, taille de buffer |
| `emission`      | `cw` / `cw_offset`, `f_offset` |
| `decimation`    | `D`, `f_max_utile`, anti-repliement |
| `clutter`       | suppression énergie statique (DC / fond) |
| `windowing`     | type de fenêtre sur le segment `n_fft` avant FFT |
| `spectrogramme` | `n_fft`, `overlap`, `skip_warmup` |
| `detection`     | bandes Fisher + ACF, `alpha`, fusion `w` |
| `affichage`     | historique score, seuil (live) |
| `bilan_liaison` | plage affichée / efficacité de bande |
| `simulation`    | respiration synthétique, SNR, clutter |

Détails dans les commentaires du YAML.

---

## Pièges courants (résumé)

1. **ADC saturé** — baisser `rx_gain` ou le `tx_gain`.
2. **Pic DC** — `cw_offset`, clutter plus agressif.
3. **Pas d’émission visible** — mise à l’échelle DAC `2**14` déjà dans `emission.py`.
4. **Aliasing après décimation** — respecter Shannon ; message d’erreur explicite possible.
5. **Module `utils` introuvable** en lançant `python utils/*.py` — exécuter depuis `MicroDopplerDetection/` ; les scripts insèrent la racine du paquet dans `sys.path` avant les imports internes.

---

## Licence

Projet interne — usage académique et recherche.
