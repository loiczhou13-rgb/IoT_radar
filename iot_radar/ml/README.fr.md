[English](README.md) | **Français**

# `iot_radar.ml` — ancien autoencodeur de spectrogrammes

> **État : ancien code, à repenser.**  Ce sous-paquet apprend encore sur les
> **spectrogrammes micro-Doppler** `.npz` enregistrés par l'ancienne chaîne.
> Cette chaîne a été supprimée : **plus rien dans le dépôt ne produit ces
> fichiers**, et l'ancien jeu de données `.npz` n'est plus conservé dans le
> dépôt.  Le code a seulement été déplacé et traduit pendant le refactor,
> pas repensé.  Il sera repensé pour prendre en entrée le **signal de phase**
> (déplacement du thorax) calculé à partir des sessions HDF5.

## Contenu

| Fichier | Rôle |
|---|---|
| `dataset.py` | `CalibrationDataset` : charge tous les `.npz` d'un dossier et découpe chaque spectrogramme en fenêtres de `n_cols` colonnes de TFCT, de forme `(1, n_fft, n_cols)`, avec le label (0 = vide, 1 = respiration) |
| `model.py` | `SpectrogramAutoencoder` : encodeur / décodeur Conv2D et tête de classification binaire sur l'espace latent |
| `train.py` | boucle d'entraînement (partie bibliothèque) : perte composite `α · reconstruction + (1 − α) · BCE`, AdamW, StepLR, meilleur checkpoint, historique CSV |

Points d'entrée : [`scripts/train.py`](../../scripts/train.py) et
[`notebooks/inference.ipynb`](../../notebooks/inference.ipynb) (courbes
d'entraînement, matrice de confusion, reconstructions).  Tous les
hyperparamètres sont dans [`configs/training.yaml`](../../configs/training.yaml).

## Lancement

Nécessite l'extra `ml` (`pip install -e ".[ml]"`) et une copie des anciens
fichiers `.npz` dans `data/ml/` (ignoré par Git) :

```
data/ml/
├── train/<n>.npz
└── val/<n>.npz      optionnel ; sinon val_split de train/
```

Chaque `.npz` contient au minimum `spectrogram_db` (`(n_frames, n_fft)`, dB),
`label`, `env`, `n_fft` et `f_s_dec_hz`.  Forme d'entrée par défaut :
`(1, 8192, 32)`, espace latent `(128, 32, 2)`.

```bash
python scripts/train.py --epochs 100
```

Options : `--config`, `--epochs`, `--batch-size`, `--lr`, `--device`
(`auto` / `cpu` / `cuda`).  Les sorties vont dans `results/` (ignoré par
Git) : `best.pt` (poids et configuration embarquée), `history.csv`,
`train.log`.

## Refonte prévue

- **Entrée :** des fenêtres du déplacement du thorax (ou de l'IQ en temps lent
  compensé) calculées par `iot_radar.pipeline` à partir des sessions HDF5, au
  lieu des spectrogrammes de l'IQ filtré passe-haut.
- **Labels :** les `/annotations` des sessions (`empty`, `breathing`,
  `apnea`, `motion`, `unknown`).
- **Données :** le script d'entraînement lit les sessions (`ReplaySource`)
  et les fait passer dans le pipeline, pour que l'entraînement voie
  exactement ce que voit la chaîne temps réel.

La règle de dépendance reste : `ml/` ne doit pas importer `acquisition/`
(vérifié par `tests/test_architecture.py`) ; c'est `scripts/train.py` qui
assemble.
