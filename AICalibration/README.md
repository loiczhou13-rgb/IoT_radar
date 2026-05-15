# AICalibration — données & jeu PyTorch

Ce dossier regroupe :

1. **`data/`** — répertoire ignoré par Git (voir `.gitignore`), utilisé en pratique comme **dépôt des enregistrements** du projet.  
2. **`dataset.py`** — charge des `.npz` au format **sessions de calibration** (`cols`, `label`, …) et expose un **`Dataset`** PyTorch.

## Deux formats de `.npz` dans le même dépôt

### A — Enregistrements « micro-Doppler supervisés » (`record_acquisition`)

Produits par :

```bash
cd MicroDopplerDetection
python utils/record_acquisition.py --subset train --env salle --label 1 --duration 120
```

Structure typique (sous `data/<train|test|val>/<index>.npz`) :

- `spectrogram_db` — colonnes du spectrogramme (dB)
- `t_wall_s`, `n_trame`, `f_hz`, métadonnées (`subset`, `sample_index`, `config_path`, `env`, `label`, IQ optionnel, etc.)

Le chemin par défaut est **`IoT_radar/AICalibration/data/`** (réglable avec `--data-root`).  
Un fichier **`.json`** auxiliaire est écrit à côté du `.npz`.

Enchaînement de plusieurs prises : **`utils/auto_record.py`** (MicroDopplerDetection).

### B — Sessions pour `CalibrationDataset` (`record_session`)

La fonction **`record_session()`** dans `dataset.py` écrit des fichiers nommés du type  
`{env}_{vide|respiration}_{timestamp}.npz` avec notamment :

| Clé | Description |
|-----|-------------|
| `cols` | `float32`, forme `(N_cols, n_fft)` — colonnes STFT (dB) |
| `label` | `0` = salle vide, `1` = présence / respiration |
| `env` | étiquette texte d’environnement |
| `f_s_dec`, `n_fft`, `hop`, `f_offset` | paramètres alignés sur `main.py` |

Ces sessions réutilisent les mêmes étapes que le flux temps réel (décimation → clutter → fenêtre → STFT), **sans** réécrire tout le schéma « dataset numéroté » ci-dessus.

### Utiliser `CalibrationDataset`

Répertoire de travail conseillé : dossier **`AICalibration/`** (pour que `import dataset` fonctionne sans paquet installé).

```bash
cd AICalibration   # depuis la racine IoT_radar
python
```

```python
from pathlib import Path
from dataset import CalibrationDataset

ds = CalibrationDataset(Path("chemin/vers/un/dossier/de/npz_format_B"))
```

Les fichiers du **format A** (`spectrogram_db`) ne sont **pas** lus tels quels par `CalibrationDataset` : il attend le format **B** (`cols`). Adapter un chargeur ou convertir les données si vous souhaitez tout fusionner.

## Dépendances & imports

Pour **`record_session()`** et la partie acquisition dans `dataset.py`, les imports pointent vers **`MicroDopplerDetection.pipeline.*`**. Il faut donc ajouter la **racine du dépôt** (`IoT_radar`) au chemin Python :

```bash
export PYTHONPATH="/chemin/absolu/vers/IoT_radar"
cd /chemin/absolu/vers/IoT_radar/AICalibration
python -c "from dataset import record_session"
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `dataset.py` | `CalibrationDataset`, `record_session()`, constantes `N_COLS`, `MIN_WINDOWS`, … |
| `model.py` | Réseau / entraînement — à compléter selon votre branche |

## Licence

Projet interne — usage académique et recherche.
