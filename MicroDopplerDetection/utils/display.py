"""
Matplotlib waterfall visualization for micro-Doppler spectrograms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from pipeline.spectrogramme import SpectrogramOutput

logger = logging.getLogger(__name__)


def _is_headless_matplotlib_backend(backend: str) -> bool:
    """
    True if Matplotlib cannot open an interactive window for ``plt.show()``.

    Do not use ``\"agg\" in backend.lower()`` — names like ``TkAgg`` and
    ``Qt5Agg`` contain the substring ``\"agg\"`` but are interactive.
    """
    b = backend.lower().strip()
    if b == "agg":
        return True
    if b in ("svg", "pdf", "ps", "template"):
        return True
    if b.startswith("module://") and "backend_agg" in b:
        if any(x in b for x in ("tkagg", "qtagg", "wx", "gtk")):
            return False
        return True
    return False


@dataclass
class DisplayConfig:
    """Subset of YAML ``affichage`` plus carrier/sample-rate metadata."""

    dynamique_dB: float
    ylim: tuple[float, float]
    colormap: str
    f_c: float
    f_s: float
    df_hz: float
    dv_mps: float


def show_waterfall(
    spec: SpectrogramOutput,
    display: DisplayConfig,
    title: str = "Micro-Doppler spectrogram",
    animate: bool = True,
    interval_ms: int = 40,
) -> None:
    """
    Plot the dB spectrogram with velocity on the vertical axis.

    Parameters
    ----------
    spec
        STFT products from :func:`pipeline.spectrogramme.compute_spectrogram`.
    display
        Visual limits, colormap, and annotation parameters.
    title
        Figure title string.
    animate
        If ``True``, progressively reveals STFT columns for a live waterfall feel;
        if ``False``, draws the full panel immediately.
    interval_ms
        Delay between animation frames when ``animate`` is ``True``.

    Returns
    -------
    None
        Blocks until the matplotlib window is closed when animation is used or
        ``plt.show()`` is called.

    Examples
    --------
    >>> # show_waterfall(spec, DisplayConfig(60, (-1,1), "inferno", 2.4e9, 200, 0.5, 0.01))

    Notes
    -----
    Physical note: the vertical axis maps Doppler frequency to radial velocity
    via ``v = f * lambda / 2``; persistent horizontal ridges at ``±fv`` indicate
    periodic chest motion despite strong static clutter at 0 Hz.
    """
    S = np.asarray(spec.S_db, dtype=np.float64)
    t = np.asarray(spec.t_s, dtype=np.float64)
    v = np.asarray(spec.v_mps, dtype=np.float64)

    vmax = float(np.nanmax(S))
    vmin = vmax - float(display.dynamique_dB)

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = (float(t[0]), float(t[-1]), float(v[0]), float(v[-1]))

    im = ax.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=display.colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_ylim(display.ylim)
    ax.set_xlabel("Slow time (s)")
    ax.set_ylabel("Doppler velocity (m/s)")
    ax.set_title(title)

    info = (
        f"f_c = {display.f_c:.3e} Hz\n"
        f"f_s (proc.) = {display.f_s:.3e} Hz\n"
        f"df ≈ {display.df_hz:.4g} Hz\n"
        f"dv ≈ {display.dv_mps:.4g} m/s"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)")

    backend = plt.get_backend()
    headless = _is_headless_matplotlib_backend(backend)
    use_anim = bool(animate) and S.shape[1] > 1 and not headless
    if use_anim:
        full = S.copy()
        mask = np.full_like(S, fill_value=vmin - 10.0)

        def _init() -> tuple:
            im.set_data(mask)
            return (im,)

        def _update(frame: int) -> tuple:
            col = min(frame, full.shape[1] - 1)
            mask[:, : col + 1] = full[:, : col + 1]
            im.set_data(mask)
            return (im,)

        anim = FuncAnimation(
            fig,
            _update,
            frames=full.shape[1],
            init_func=_init,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        fig._microdoppler_anim = anim  # prevent GC before show
        logger.debug("Started waterfall animation (%d frames)", full.shape[1])
    else:
        if animate and headless:
            logger.info("Static plot (non-interactive backend; animation disabled).")
        im.set_data(S)

    plt.tight_layout()
    if headless:
        plt.close(fig)
    else:
        plt.show()


def save_waterfall_png(
    spec: SpectrogramOutput,
    display: DisplayConfig,
    path: str,
    title: str = "Micro-Doppler spectrogram",
) -> None:
    """
    Save a static spectrogram image without opening an interactive window.

    Parameters
    ----------
    spec
        STFT products.
    display
        Visual configuration.
    path
        Output ``.png`` path.
    title
        Figure title.

    Notes
    -----
    Physical note: identical mapping as :func:`show_waterfall` but suited for
    headless Raspberry Pi logging when ``MPLBACKEND=Agg``.
    """
    S = np.asarray(spec.S_db, dtype=np.float64)
    t = np.asarray(spec.t_s, dtype=np.float64)
    v = np.asarray(spec.v_mps, dtype=np.float64)
    vmax = float(np.nanmax(S))
    vmin = vmax - float(display.dynamique_dB)

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = (float(t[0]), float(t[-1]), float(v[0]), float(v[-1]))
    im = ax.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=display.colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_ylim(display.ylim)
    ax.set_xlabel("Slow time (s)")
    ax.set_ylabel("Doppler velocity (m/s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax).set_label("Magnitude (dB)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved spectrogram to %s", path)
