# IoT Radar — micro-Doppler & calibration

Dépôt pour un dispositif radar **onde continue (CW)** destiné à la **détection de signes de vie** (respiration) en contexte catastrophe, à partir d’un **micro-Doppler** spectral, avec volet **acquisition de données** et **calibration par apprentissage**.

## Rôle des dossiers

| Dossier | Contenu |
|---------|---------|
| [**MicroDopplerDetection/**](MicroDopplerDetection/README.md) | Chaîne temps réel : PlutoSDR ou simulation, décimation, suppression de clutter, **fenêtrage**, STFT, détection (Fisher × ACF), dashboard. Scripts d’**enregistrement** et de **relecture**. |
| [**AICalibration/**](AICalibration/README.md) | Données labellisées par défaut sous `data/`, jeu **PyTorch** `CalibrationDataset`, utilitaire `record_session()` (format `.npz` « colonnes STFT »). |

La racine des enregistrements **numérotés** (train / test / val) produits par `MicroDopplerDetection/utils/record_acquisition.py` est par défaut :

`AICalibration/data/` (voir `MicroDopplerDetection/utils/repo_paths.py`).

## Matériel visé

- **ADALM-PLUTO** (AD9363), libiio / `pyadi-iio`
- Pilotage typique depuis une machine Linux (dont **WSL2**) avec accès USB au Pluto

## Prérequis logiciels (résumé)

```bash
sudo apt update
sudo apt install -y libiio-dev libiio-utils git python3 python3-venv python3-pip build-essential
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pyyaml pyadi-iio
pip install torch   # uniquement pour AICalibration / PyTorch
```

Pour le Pluto : `iio_info -s` doit lister la carte.

## Lancement rapide (sans matériel)

À partir de la racine du dépôt :

```bash
export PYTHONPATH="$(pwd)"
python -m MicroDopplerDetection.main --simulation
```

Ou :

```bash
cd MicroDopplerDetection
python main.py --simulation
```

Documentation détaillée du pipeline et des options : [**MicroDopplerDetection/README.md**](MicroDopplerDetection/README.md).  
Jeux de données et autoencodeur de calibration : [**AICalibration/README.md**](AICalibration/README.md).

## Licence

Projet interne — usage académique et recherche.
