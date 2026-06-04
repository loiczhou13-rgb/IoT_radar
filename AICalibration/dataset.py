"""Calibration dataset for the supervised autoencoder.

Source de vérité pour les enregistrements
-----------------------------------------
Les fichiers ``.npz`` sont produits par
``MicroDopplerDetection/utils/record_acquisition.py`` (et son lanceur
``auto_record.py``).  Arborescence typique ::

    AICalibration/data/
    ├── train/<n>.npz
    ├── test/<n>.npz
    └── val/<n>.npz

Chaque ``.npz`` contient au minimum :

    - ``spectrogram_db`` : float64, forme ``(N_frames, n_fft)``
                            — colonnes STFT successives (en dB)
    - ``label``          : scalaire ``int8`` (0 = vide, 1 = respiration)
    - ``env``            : str        (étiquette d'environnement)
    - ``n_fft``          : int        (largeur fréquentielle)
    - ``f_s_dec_hz``     : float      (Hz)
    - métadonnées : ``subset``, ``sample_index``, ``config_path``,
      ``data_root``, ``utc_finished``, ``n_trame``, ``t_wall_s``, etc.

CalibrationDataset
------------------
``Dataset`` PyTorch qui :

1. Charge récursivement tous les ``.npz`` sous ``data_dir``.
2. Découpe chaque enregistrement en fenêtres glissantes de ``N_COLS``
   colonnes STFT (le long de l'axe temporel).
3. Optionnellement normalise chaque fenêtre (zéro-mean / unit-var).
4. Renvoie ``(tensor[1, n_fft, N_COLS], label)`` — convention PyTorch
   ``(C, H, W)`` avec ``H = n_fft`` (fréquence) et ``W = N_COLS`` (temps).

Le format ``(1, n_fft, N_COLS)`` est conçu pour un autoencodeur **Conv2D**
(encoder + decoder), qui exploite la localité spectro-temporelle.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Nombre de colonnes STFT par fenêtre d'entrée.
#: Avec la config par défaut (n_fft=8192, overlap=0.5, f_s_dec≈2 kHz),
#: une colonne ≈ 0.45 s, donc 32 colonnes ≈ 14 s — couvre plusieurs
#: cycles respiratoires (~0.1–0.5 Hz).
N_COLS: int = 32

#: Avertissement si un enregistrement fournit moins de fenêtres que cela.
MIN_WINDOWS: int = 10


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CalibrationDataset(Dataset):
    """Sliding-window STFT dataset for autoencoder training.

    Parameters
    ----------
    data_dir : str or Path
        Répertoire scanné récursivement (``rglob("*.npz")``).  Pointer
        sur ``AICalibration/data`` pour tout charger, ou sur
        ``AICalibration/data/train`` pour un seul split.
    n_cols : int, optional
        Nombre de colonnes STFT par fenêtre.  Défaut :data:`N_COLS`.
    stride : int, optional
        Pas entre deux fenêtres successives extraites d'un même
        enregistrement.  Par défaut ``n_cols`` (fenêtres disjointes).
        Un ``stride`` plus petit augmente la quantité d'exemples au prix
        d'une forte corrélation entre fenêtres voisines.
    normalise : bool, optional
        Si ``True`` (défaut) chaque fenêtre est ramenée à moyenne nulle
        et écart-type unité (sur l'ensemble ``n_fft × n_cols``).  Cela
        retire la dépendance au niveau absolu de puissance (distance, RX
        gain).

    Notes
    -----
    Tous les enregistrements sous ``data_dir`` doivent partager le même
    ``n_fft`` ; une ``ValueError`` est levée sinon.

    Le jeu est entièrement chargé en RAM à la construction.  Pour des
    enregistrements de 2 min avec ``n_fft=8192`` (~16 Mio par fichier),
    quelques dizaines de prises tiennent sans difficulté.
    """

    def __init__(
        self,
        data_dir: str | Path,
        n_cols: int = N_COLS,
        stride: int | None = None,
        normalise: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_cols = int(n_cols)
        self.stride = int(stride) if stride is not None else self.n_cols
        self.normalise = bool(normalise)

        if self.n_cols <= 0:
            raise ValueError("n_cols doit être > 0.")
        if self.stride <= 0:
            raise ValueError("stride doit être > 0.")

        self._windows: list[np.ndarray] = []  # chaque entrée : (n_fft, n_cols) float32
        self._labels: list[int] = []

        self._load_all()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Répertoire de données introuvable : {self.data_dir}"
            )

        npz_files = sorted(self.data_dir.rglob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"Aucun fichier .npz trouvé sous {self.data_dir}. "
                "Lancer d'abord MicroDopplerDetection/utils/record_acquisition.py."
            )

        n_fft_ref: int | None = None
        total_windows = 0

        for path in npz_files:
            with np.load(path, allow_pickle=False) as data:
                if "spectrogram_db" not in data.files:
                    logger.warning(
                        "%s ignoré — pas de clé 'spectrogram_db' "
                        "(enregistrement fait avec --no-spectrogram ?).",
                        path.name,
                    )
                    continue
                if "label" not in data.files:
                    logger.warning("%s ignoré — pas de clé 'label'.", path.name)
                    continue

                spec = np.asarray(data["spectrogram_db"], dtype=np.float32)
                label = int(np.asarray(data["label"]).item())
                env = (
                    str(np.asarray(data["env"]).item())
                    if "env" in data.files else "?"
                )

            if spec.ndim != 2:
                logger.warning(
                    "%s ignoré — spectrogram_db de dimension %d (attendu 2).",
                    path.name, spec.ndim,
                )
                continue

            # spectrogram_db : (N_frames, n_fft) — on transpose pour découper
            # le long de l'axe temporel.
            spec_t = spec.T  # (n_fft, N_frames)
            n_fft_file, n_total = spec_t.shape

            if n_fft_ref is None:
                n_fft_ref = n_fft_file
            elif n_fft_file != n_fft_ref:
                raise ValueError(
                    f"n_fft incohérent dans {path.name} : "
                    f"attendu {n_fft_ref}, trouvé {n_fft_file}."
                )

            if n_total < self.n_cols:
                logger.warning(
                    "%s ignoré — trop court (%d colonnes < n_cols=%d)",
                    path.name, n_total, self.n_cols,
                )
                continue

            windows_in_file = 0
            for start in range(0, n_total - self.n_cols + 1, self.stride):
                window = spec_t[:, start: start + self.n_cols].copy()  # (n_fft, n_cols)
                self._windows.append(window)
                self._labels.append(label)
                windows_in_file += 1

            if windows_in_file < MIN_WINDOWS:
                logger.warning(
                    "%s — seulement %d fenêtres extraites (MIN_WINDOWS=%d). "
                    "Envisager un enregistrement plus long.",
                    path.name, windows_in_file, MIN_WINDOWS,
                )

            total_windows += windows_in_file
            logger.info(
                "Chargé %s — env='%s', label=%d, %d colonnes → %d fenêtres",
                path.name, env, label, n_total, windows_in_file,
            )

        if not self._windows:
            raise RuntimeError(
                "Aucune fenêtre extraite. Vérifier la durée des enregistrements "
                f"(n_cols={self.n_cols}, stride={self.stride})."
            )

        n1 = sum(1 for lbl in self._labels if lbl == 1)
        n0 = len(self._labels) - n1
        logger.info(
            "Dataset prêt — %d fenêtres total (label=0: %d, label=1: %d, "
            "ratio 1/0=%.2f)",
            total_windows, n0, n1, n1 / max(n0, 1),
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a normalised spectrogram window and its label.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(x, y)`` avec ``x`` de forme ``(1, n_fft, n_cols)`` (float32)
            et ``y`` scalaire ``int64`` (0 ou 1).
        """
        window = self._windows[idx].astype(np.float32, copy=True)  # (n_fft, n_cols)

        if self.normalise:
            mu = float(window.mean())
            sigma = float(window.std())
            if sigma > 1e-6:
                window = (window - mu) / sigma
            else:
                window = window - mu

        x = torch.from_numpy(window).unsqueeze(0)              # (1, n_fft, n_cols)
        y = torch.tensor(self._labels[idx], dtype=torch.long)  # scalaire
        return x, y

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def n_fft(self) -> int:
        """Dimension fréquentielle (``H``) commune à toutes les fenêtres."""
        return self._windows[0].shape[0]

    @property
    def class_weights(self) -> torch.Tensor:
        """Poids ``[w0, w1]`` inverse-fréquence pour ``CrossEntropyLoss``.

        À passer à ``torch.nn.CrossEntropyLoss(weight=dataset.class_weights)``
        si les classes sont déséquilibrées.
        """
        n_total = len(self._labels)
        n1 = sum(self._labels)
        n0 = n_total - n1
        w0 = n_total / (2.0 * max(n0, 1))
        w1 = n_total / (2.0 * max(n1, 1))
        return torch.tensor([w0, w1], dtype=torch.float32)

    def summary(self) -> str:
        """Résumé lisible du dataset."""
        n1 = sum(self._labels)
        n0 = len(self._labels) - n1
        return (
            f"CalibrationDataset — {len(self)} fenêtres "
            f"(n_fft={self.n_fft}, n_cols={self.n_cols}, stride={self.stride})\n"
            f"  label=0 (vide)        : {n0}\n"
            f"  label=1 (respiration) : {n1}\n"
            f"  ratio 1/0             : {n1 / max(n0, 1):.2f}\n"
            f"  data_dir              : {self.data_dir}"
        )
