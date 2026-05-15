"""Calibration dataset for the supervised autoencoder.

Acquisition procedure
---------------------
For each environment (label), run two recording sessions with the Pluto:

1. **Salle vide** (label=0) — no one in the radar field of view.
   Recommended duration: 5 min minimum per environment.
   Environments to cover: chambre_reverb, couloir, exterieur, decombres_sim, ...

2. **Personne immobile qui respire normalement** (label=1) — one person at rest,
   at the expected operating distance (1–5 m), oriented face-on to the antenna.
   Same duration as the empty-room session for class balance.

Each session is recorded via ``record_session()`` which runs the full
pipeline (decimation → clutter suppression → STFT) and saves the resulting
STFT columns to a compressed ``.npz`` file.  Raw IQ is NOT saved — only
the post-pipeline ``col_db`` columns (``float32``, shape ``(n_fft,)``) and
their metadata.

Dataset structure on disk
--------------------------
::

    AICalibration/
    └── data/
        ├── chambre_reverb_vide_20240601_143000.npz
        ├── chambre_reverb_respiration_20240601_144000.npz
        ├── exterieur_vide_20240601_150000.npz
        └── ...

Each ``.npz`` contains:
    - ``cols``      : float32 array, shape ``(N_cols, n_fft)``
    - ``label``     : int scalar  (0 = vide, 1 = respiration)
    - ``env``       : str         (environment tag)
    - ``f_s_dec``   : float       (decimated sampling rate, Hz)
    - ``n_fft``     : int
    - ``hop``       : int
    - ``f_offset``  : float       (baseband offset, Hz)
    - ``timestamp`` : str         (ISO-8601)

CalibrationDataset
------------------
PyTorch Dataset that:
  1. Loads all ``.npz`` files from a directory.
  2. Slices each recording into non-overlapping windows of ``N_COLS`` columns.
  3. Normalises each window to zero mean and unit variance.
  4. Returns ``(tensor[1, n_fft, N_COLS], label)`` — ready for a 2-D CNN.

The single channel dimension follows the PyTorch ``(C, H, W)`` convention
where H = frequency bins (n_fft) and W = time columns (N_COLS).
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match pipeline config
# ---------------------------------------------------------------------------

#: Number of STFT columns per input window.
#: At hop=n_fft/2=96, f_s_dec≈100 Hz → one column every ~1 s.
#: 32 columns ≈ 32 s → covers ~3–10 breathing cycles at 0.1–0.3 Hz.
N_COLS: int = 32

#: Minimum number of windows required per recording to be included.
MIN_WINDOWS: int = 10


# ---------------------------------------------------------------------------
# Acquisition helper — records one session and saves to .npz
# ---------------------------------------------------------------------------

def record_session(
    cfg: dict[str, Any],
    env_label: str,
    presence: bool,
    output_dir: str | Path,
    duration_s: float = 300.0,
    simulation: bool = False,
) -> Path:
    """Run the pipeline and save STFT columns for one calibration session.

    This function reuses the exact same pipeline stages as ``main.py``
    (decimation → clutter → STFT) so that the saved columns are
    representative of what the model will see at inference.

    Parameters
    ----------
    cfg : dict
        Full YAML configuration dictionary (same as ``main.py``).
    env_label : str
        Short tag for the environment, e.g. ``"chambre_reverb"``,
        ``"exterieur"``, ``"couloir"``.  Used in the filename and stored
        as metadata.  No spaces — use underscores.
    presence : bool
        ``True`` if a person is breathing in the scene (label=1),
        ``False`` for an empty room (label=0).
    output_dir : str or Path
        Directory where the ``.npz`` file will be saved.
        Created automatically if it does not exist.
    duration_s : float, optional
        Recording duration in seconds.  Default 120 s (2 min).
    simulation : bool, optional
        Use ``stream_simulation`` instead of the PlutoSDR.  Default False.

    Returns
    -------
    Path
        Path to the saved ``.npz`` file.

    Notes
    -----
    The warm-up frames (clutter filter transient) are discarded exactly
    as in ``main.py`` — the saved columns reflect the steady-state
    pipeline output only.

    The function prints a progress line every 10 % of the requested
    duration so the operator knows the recording is running.
    """
    import math
    from MicroDopplerDetection.pipeline.decimation import Decimator
    from MicroDopplerDetection.pipeline.clutter import ClutterFilter
    from MicroDopplerDetection.pipeline.spectrogramme import compute_single_column
    from MicroDopplerDetection.pipeline.windowing import get_window
    from MicroDopplerDetection.pipeline.emission import generate_tx_buffer
    from MicroDopplerDetection.pipeline.acquisition import stream_pluto, stream_simulation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sdr_cfg = cfg["sdr"]
    dec_cfg = cfg["decimation"]
    clu_cfg = cfg["clutter"]
    spec_cfg = cfg["spectrogramme"]
    win_cfg = cfg["windowing"]
    emi = cfg.get("emission", {})

    f_s = float(sdr_cfg["f_s"])
    f_c = float(sdr_cfg["f_c"])
    n_fft = int(spec_cfg["n_fft"])
    overlap = float(spec_cfg["overlap"])
    hop = max(1, int(n_fft * (1.0 - overlap)))

    do_decimate = dec_cfg.get("enable", True)
    D = int(dec_cfg["D"]) if do_decimate else 1
    f_max_utile = float(dec_cfg.get("f_max_utile", 10.0))

    f_offset = 0.0
    if emi.get("mode") == "cw_offset":
        f_offset = float(emi.get("f_offset", 0.0))

    decimator = Decimator(f_s=f_s, D=D, f_max_utile=f_max_utile)
    f_s_dec = decimator.f_s_out

    clutter_filter = ClutterFilter(
        mode=clu_cfg["mode"],
        fs=f_s_dec,
        alpha=clu_cfg.get("alpha", 0.9999),
        butterworth_order=clu_cfg.get("butterworth_order", 2),
        butterworth_cutoff=clu_cfg.get("butterworth_cutoff", 0.05),
    )
    window = get_window(win_cfg["mode"], n_fft)

    # Warm-up: same logic as main.py _auto_skip_warmup
    mode_clu = clu_cfg.get("mode", "butterworth")
    if mode_clu in ("iir", "mean"):
        alpha_clu = float(clu_cfg.get("alpha", 0.9999))
        tau_s = 1.0 / max((1.0 - alpha_clu) * f_s_dec, 1e-12)
    elif mode_clu == "butterworth":
        f_cut = float(clu_cfg.get("butterworth_cutoff", 0.05))
        tau_s = 1.0 / (2.0 * math.pi * max(f_cut, 1e-6))
    else:
        tau_s = 0.0
    hop_s = hop / f_s_dec
    skip_warmup = int(math.ceil(3.0 * tau_s / hop_s)) if hop_s > 0 else 0
    skip_warmup = spec_cfg.get("skip_warmup", skip_warmup)

    # Total columns to collect (excluding warm-up)
    n_cols_target = int(math.ceil(duration_s / hop_s))
    progress_step = max(1, n_cols_target // 10)

    # Build IQ stream
    sim_cfg = cfg.get("simulation", {})
    if simulation or sim_cfg.get("enable", False):
        iq_stream = stream_simulation(
            f_c=f_c,
            f_s=f_s,
            buffer_size=sdr_cfg["buffer_size"],
            fv=sim_cfg.get("fv", 0.3),
            D_mm=sim_cfg.get("D_mm", 4.0),
            snr_dB=sim_cfg.get("snr_dB", 10.0),
            f_offset=f_offset,
            clutter_amplitude=sim_cfg.get("clutter_amplitude", 100.0),
        )
    else:
        tx_buffer = generate_tx_buffer(
            mode=emi.get("mode", "cw"),
            buffer_size=sdr_cfg["buffer_size"],
            f_s=f_s,
            f_offset=f_offset,
        )
        iq_stream = stream_pluto(
            uri=sdr_cfg["uri"],
            f_c=f_c,
            f_s=f_s,
            rx_gain=sdr_cfg["rx_gain"],
            tx_gain=sdr_cfg["tx_gain"],
            buffer_size=sdr_cfg["buffer_size"],
            tx_buffer=tx_buffer,
        )

    cols: list[np.ndarray] = []
    ring: deque = deque(maxlen=n_fft)
    frame_counter = 0
    samples_since_last_fft = 0

    logger.info(
        "Enregistrement — env='%s', label=%d, durée=%.0f s, "
        "n_cols_cible=%d, skip_warmup=%d",
        env_label, int(presence), duration_s, n_cols_target, skip_warmup,
    )

    for raw_buf in iq_stream:
        iq_dec = decimator(raw_buf)
        iq_filt = clutter_filter(iq_dec)

        ring.extend(iq_filt)
        samples_since_last_fft += len(iq_filt)

        while len(ring) == n_fft and samples_since_last_fft >= hop:
            samples_since_last_fft -= hop
            frame_counter += 1

            if frame_counter <= skip_warmup:
                continue

            segment = np.fromiter(ring, dtype=np.complex64, count=n_fft)
            col = compute_single_column(segment, f_s_dec, f_c, window)
            cols.append(col.col_db.astype(np.float32))

            n_collected = len(cols)
            if n_collected % progress_step == 0:
                pct = 100 * n_collected / n_cols_target
                logger.info("  %.0f %% (%d / %d colonnes)", pct, n_collected, n_cols_target)

            if n_collected >= n_cols_target:
                break

        if len(cols) >= n_cols_target:
            break

    cols_array = np.stack(cols, axis=0)  # (N_cols_total, n_fft)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    label_str = "respiration" if presence else "vide"
    filename = f"{env_label}_{label_str}_{timestamp}.npz"
    out_path = output_dir / filename

    np.savez_compressed(
        out_path,
        cols=cols_array,
        label=np.int8(presence),
        env=np.str_(env_label),
        f_s_dec=np.float32(f_s_dec),
        n_fft=np.int32(n_fft),
        hop=np.int32(hop),
        f_offset=np.float32(f_offset),
        timestamp=np.str_(timestamp),
    )

    logger.info(
        "Session sauvegardée → %s (%d colonnes, label=%d)",
        out_path, len(cols_array), int(presence),
    )
    return out_path


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CalibrationDataset(Dataset):
    """Sliding-window STFT dataset for supervised AE training.

    Each item is a normalised spectrogram window of shape
    ``(1, n_fft, N_COLS)`` and an integer label (0 or 1).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``.npz`` calibration files produced by
        :func:`record_session`.
    n_cols : int, optional
        Number of STFT columns per window.  Default :data:`N_COLS` (32).
    stride : int, optional
        Step between successive windows extracted from the same recording.
        ``stride=n_cols`` → non-overlapping windows (default).
        ``stride=1`` → maximum overlap (data augmentation, but high
        correlation between adjacent samples — use with care).
    normalise : bool, optional
        If ``True`` (default), each window is normalised to zero mean and
        unit variance across all ``(n_fft × N_COLS)`` values.  This
        removes the dependency on absolute power level (distance, RX gain).

    Notes
    -----
    All recordings in *data_dir* must share the same ``n_fft`` value.
    A ``ValueError`` is raised at construction time if inconsistent
    ``n_fft`` values are found across files.

    The dataset is fully loaded into RAM at construction time.  For
    typical recordings (5 min × 2 labels × 5 environments at ~1 col/s)
    the total size is modest: 300 cols × 10 sessions × 8192 bins × 4 B
    ≈ 100 MB — well within normal workstation RAM.
    """

    def __init__(
        self,
        data_dir: str | Path,
        n_cols: int = N_COLS,
        stride: int | None = None,
        normalise: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_cols = n_cols
        self.stride = stride if stride is not None else n_cols
        self.normalise = normalise

        self._windows: list[np.ndarray] = []  # each: (n_fft, n_cols) float32
        self._labels: list[int] = []

        self._load_all()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        npz_files = sorted(self.data_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"Aucun fichier .npz trouvé dans {self.data_dir}. "
                f"Lancer record_session() d'abord."
            )

        n_fft_ref: int | None = None
        total_windows = 0

        for path in npz_files:
            data = np.load(path, allow_pickle=True)
            cols: np.ndarray = data["cols"]          # (N, n_fft) float32
            label: int = int(data["label"])
            n_fft_file: int = int(data["n_fft"])
            env: str = str(data["env"])

            if n_fft_ref is None:
                n_fft_ref = n_fft_file
            elif n_fft_file != n_fft_ref:
                raise ValueError(
                    f"n_fft incohérent dans {path.name} : "
                    f"attendu {n_fft_ref}, trouvé {n_fft_file}."
                )

            # cols shape is (N_total, n_fft) — transpose to (n_fft, N_total)
            # for slicing along the time axis
            cols_t = cols.T  # (n_fft, N_total)
            n_total = cols_t.shape[1]

            if n_total < self.n_cols:
                logger.warning(
                    "%s ignoré — trop court (%d colonnes < n_cols=%d)",
                    path.name, n_total, self.n_cols,
                )
                continue

            windows_in_file = 0
            for start in range(0, n_total - self.n_cols + 1, self.stride):
                window = cols_t[:, start: start + self.n_cols]  # (n_fft, n_cols)
                self._windows.append(window.copy())
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

        n1 = sum(1 for l in self._labels if l == 1)
        n0 = len(self._labels) - n1
        logger.info(
            "Dataset prêt — %d fenêtres total (label=0: %d, label=1: %d, "
            "ratio=%.2f)",
            total_windows, n0, n1,
            n1 / max(n0, 1),
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
            ``(x, y)`` where ``x`` has shape ``(1, n_fft, n_cols)``
            (float32) and ``y`` is a scalar int64 label (0 or 1).
        """
        window = self._windows[idx].astype(np.float32)  # (n_fft, n_cols)

        if self.normalise:
            mu = window.mean()
            sigma = window.std()
            if sigma > 1e-6:
                window = (window - mu) / sigma
            else:
                window = window - mu

        # Add channel dim: (1, n_fft, n_cols)
        x = torch.from_numpy(window).unsqueeze(0)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def n_fft(self) -> int:
        """Frequency dimension of each window."""
        return self._windows[0].shape[0]

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced datasets.

        Pass to ``torch.nn.CrossEntropyLoss(weight=dataset.class_weights)``.
        """
        n_total = len(self._labels)
        n1 = sum(self._labels)
        n0 = n_total - n1
        w0 = n_total / (2.0 * max(n0, 1))
        w1 = n_total / (2.0 * max(n1, 1))
        return torch.tensor([w0, w1], dtype=torch.float32)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        n1 = sum(self._labels)
        n0 = len(self._labels) - n1
        return (
            f"CalibrationDataset — {len(self)} fenêtres "
            f"(n_fft={self.n_fft}, n_cols={self.n_cols}, stride={self.stride})\n"
            f"  label=0 (vide)        : {n0}\n"
            f"  label=1 (respiration) : {n1}\n"
            f"  ratio 1/0             : {n1/max(n0,1):.2f}\n"
            f"  data_dir              : {self.data_dir}"
        )