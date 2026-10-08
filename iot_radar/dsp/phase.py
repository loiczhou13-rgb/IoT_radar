"""DC-offset compensation and phase demodulation of a slow-time IQ sequence.

Place in the chain: after the down-conversion to slow time (≈ 20 Hz), before
the displacement analysis.  The slow-time signal of a CW radar is::

    x[n] = C + A · exp(j(θ0 − 4π d[n] / λ)) + w[n]

In the IQ plane the samples lie on an **arc of circle** of radius ``A``
centred on the DC offset ``C`` (TX→RX leakage + static clutter + receiver DC).
The chest displacement ``d`` is carried by the angle *around C*, not around
the origin, hence the order of operations:

1. estimate ``C`` (circle fitting, :func:`fit_circle`) and subtract it;
2. extract the angle (arctangent + unwrapping, or DACM);
3. convert to displacement ``d = −λ/(4π) · φ`` (:func:`phase_to_displacement_m`).

.. warning::
   Removing ``C`` with a **high-pass filter** is *not* equivalent: the
   high-pass also removes the mean of the useful term ``A·exp(jφ)``, so the
   remaining vector no longer rotates around the origin and its angle is
   wrong (bug B3 of the former chain).  High-pass clutter removal is only
   acceptable for magnitude products such as a spectrogram.

Reference: G. Paterniani *et al.*, "Radar-based monitoring of vital signs: a
tutorial overview", Proc. IEEE 111(3), 2023.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

VALID_DC_METHODS = ("circle", "mean", "none")
VALID_DEMODULATIONS = ("arctan", "dacm", "linear")


# ---------------------------------------------------------------------------
# Circle fitting (DC offset compensation)
# ---------------------------------------------------------------------------

@dataclass
class CircleFit:
    """Circle fitted to IQ samples.

    Attributes
    ----------
    center : complex
        Estimated DC offset ``C`` (centre of the arc), in IQ units.
    radius : float
        Estimated echo amplitude ``A``, in IQ units.
    rms_residual : float
        RMS distance of the samples to the circle, divided by the radius
        (dimensionless; small is good).
    arc_span_rad : float
        Angle covered by the samples around ``center`` (rad).
    valid : bool
        ``True`` if the fit is considered reliable (see :func:`fit_circle`).
    """

    center: complex
    radius: float
    rms_residual: float
    arc_span_rad: float
    valid: bool


def _arc_span_rad(angles_rad: np.ndarray) -> float:
    """Angle covered by a set of angles: 2π minus the largest circular gap."""
    if angles_rad.size < 2:
        return 0.0
    sorted_angles = np.sort(np.mod(angles_rad, 2.0 * np.pi))
    wrapped = np.concatenate((sorted_angles, [sorted_angles[0] + 2.0 * np.pi]))
    largest_gap = np.diff(wrapped).max()
    return float(2.0 * np.pi - largest_gap)


def fit_circle(
    z: np.ndarray,
    n_iterations: int = 20,
    min_arc_rad: float = 0.3,
    max_relative_residual: float = 0.3,
) -> CircleFit:
    """Fit a circle to complex samples (algebraic start + geometric refinement).

    The algebraic Kåsa fit (linear least squares on
    ``x² + y² + D·x + E·y + F = 0``) gives the starting point, refined by
    Gauss–Newton iterations on the geometric distance ``|z − c| − r``.  The
    data are centred and scaled first, for numerical conditioning.

    Parameters
    ----------
    z : numpy.ndarray
        Complex IQ samples, 1-D (any unit).
    n_iterations : int, optional
        Maximum number of Gauss–Newton iterations.
    min_arc_rad : float, optional
        Minimum arc span for a valid fit.  A short arc (chest excursion small
        compared with λ, or noise only) leaves the centre ill-determined.
    max_relative_residual : float, optional
        Maximum RMS residual (relative to the radius) for a valid fit.  A
        circle fitted on a pure noise cloud has a relative residual of ≈ 0.4;
        accepting it would turn noise into a random-walk phase whose red
        spectrum looks like breathing.  0.3 ≈ an arc SNR of 10 dB.

    Returns
    -------
    CircleFit
        ``valid`` is ``False`` when the fit is unreliable; the caller should
        then use :func:`linear_demodulation`.
    """
    z = np.asarray(z, dtype=np.complex128)
    if z.size < 3:
        return CircleFit(complex(np.mean(z)) if z.size else 0j, 0.0, np.inf, 0.0, False)

    mean = z.mean()
    scale = float(np.sqrt(np.mean(np.abs(z - mean) ** 2)))
    if scale <= 0.0 or not np.isfinite(scale):
        return CircleFit(complex(mean), 0.0, np.inf, 0.0, False)
    u = (z - mean) / scale
    x, y = u.real, u.imag

    # --- Algebraic (Kåsa) fit: x² + y² + D·x + E·y + F = 0 ------------------
    design = np.column_stack((x, y, np.ones_like(x)))
    rhs = -(x * x + y * y)
    (coef_d, coef_e, coef_f), *_ = np.linalg.lstsq(design, rhs, rcond=None)
    center_x, center_y = -coef_d / 2.0, -coef_e / 2.0
    radius_squared = center_x * center_x + center_y * center_y - coef_f
    if not np.isfinite(radius_squared) or radius_squared <= 0.0:
        return CircleFit(complex(mean), 0.0, np.inf, 0.0, False)
    params = np.array([center_x, center_y, np.sqrt(radius_squared)])

    # --- Geometric refinement (Gauss–Newton on |z − c| − r) ----------------
    for _ in range(n_iterations):
        dx, dy = x - params[0], y - params[1]
        distance = np.hypot(dx, dy)
        distance = np.where(distance < 1e-12, 1e-12, distance)
        residual = distance - params[2]
        jacobian = np.column_stack((-dx / distance, -dy / distance, -np.ones_like(distance)))
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        params = params + step
        if np.linalg.norm(step) < 1e-9 * max(1.0, abs(params[2])):
            break

    center_x, center_y, radius = params
    return _circle_fit_result(x, y, center_x, center_y, radius, mean, scale, min_arc_rad, max_relative_residual)


def _circle_fit_result(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    mean: complex,
    scale: float,
    min_arc_rad: float,
    max_relative_residual: float,
) -> CircleFit:
    """Residual, arc span and validity of a circle fitted on normalised samples ``(x, y)``."""
    if not np.all(np.isfinite([center_x, center_y, radius])) or radius <= 0.0:
        return CircleFit(complex(mean), 0.0, np.inf, 0.0, False)
    residual = np.hypot(x - center_x, y - center_y) - radius
    relative_residual = float(np.sqrt(np.mean(residual * residual)) / radius)
    span_rad = _arc_span_rad(np.arctan2(y - center_y, x - center_x))

    center = complex(mean + scale * (center_x + 1j * center_y))
    valid = bool(span_rad >= min_arc_rad and relative_residual <= max_relative_residual)
    return CircleFit(center, float(radius * scale), relative_residual, span_rad, valid)


def fit_circle_pratt(z: np.ndarray, min_arc_rad: float = 0.3, max_relative_residual: float = 0.3) -> CircleFit:
    """Fit a circle with the algebraic method of Pratt (non-iterative).

    Minimises the algebraic distance ``A(x² + y²) + Bx + Cy + D`` under
    Pratt's normalisation ``B² + C² − 4AD = 1``, which removes the bias of
    the Kåsa fit on short arcs (V. Pratt, "Direct least-squares fitting of
    algebraic surfaces", Computer Graphics 21(4), 1987), solved with the
    SVD formulation of N. Chernov.  It is one of the centre estimators
    compared by M. Zakrzewski, H. Raittinen and J. Vanhala, "Comparison of
    center estimation algorithms for heart and respiration monitoring with
    microwave Doppler radar", IEEE Sensors J. 12(3), 2012.  On breathing arcs
    of 0.5 rad or more it is as accurate as the geometric fit and 3 to 7
    times faster (MOTION_ROBUSTNESS_PLAN.md §3.3).

    Parameters and return value as :func:`fit_circle`.
    """
    z = np.asarray(z, dtype=np.complex128)
    if z.size < 4:
        return CircleFit(complex(np.mean(z)) if z.size else 0j, 0.0, np.inf, 0.0, False)
    mean = z.mean()
    scale = float(np.sqrt(np.mean(np.abs(z - mean) ** 2)))
    if scale <= 0.0 or not np.isfinite(scale):
        return CircleFit(complex(mean), 0.0, np.inf, 0.0, False)
    u = (z - mean) / scale
    x, y = u.real, u.imag

    design = np.column_stack((x * x + y * y, x, y, np.ones_like(x)))
    _, singular_values, vt = np.linalg.svd(design, full_matrices=False)
    v = vt.T
    if singular_values[3] / singular_values[0] < 1e-12:   # samples exactly on a circle
        coefficients = v[:, 3]
    else:
        w = v @ np.diag(singular_values) @ v.T
        pratt_inverse = np.array([[0.0, 0.0, 0.0, -0.5], [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0], [-0.5, 0.0, 0.0, 0.0]])
        eigenvalues, eigenvectors = np.linalg.eig(w @ pratt_inverse @ w)
        eigenvalues = np.real(eigenvalues)
        smallest_positive = int(np.argmin(np.where(eigenvalues > 0, eigenvalues, np.inf)))
        coefficients = np.linalg.solve(w, np.real(eigenvectors[:, smallest_positive]))
    a, b, c, d = coefficients
    if a == 0.0:
        return CircleFit(complex(mean), 0.0, np.inf, 0.0, False)
    center_x, center_y = -b / (2.0 * a), -c / (2.0 * a)
    radius = float(np.sqrt(max(b * b + c * c - 4.0 * a * d, 0.0)) / (2.0 * abs(a)))
    return _circle_fit_result(x, y, center_x, center_y, radius, mean, scale, min_arc_rad, max_relative_residual)


@dataclass
class CenterTrack:
    """Centre of the IQ arc estimated on sliding sub-windows (M2).

    Attributes
    ----------
    times_s : numpy.ndarray
        Middle of each sub-window (s from the start of the window).
    centers : numpy.ndarray
        Centre fitted on each sub-window (complex).
    valid : numpy.ndarray
        ``True`` for the sub-windows whose fit is valid (arc, residual).
    center_per_sample : numpy.ndarray
        Centre at every sample of the window, linearly interpolated between
        the valid sub-windows and constant beyond them (complex, ``(N,)``).
    """

    times_s: np.ndarray
    centers: np.ndarray
    valid: np.ndarray
    center_per_sample: np.ndarray


def track_center(
    z: np.ndarray,
    f_s_hz: float,
    window_s: float = 10.0,
    step_s: float = 1.0,
    method: str = "pratt",
    min_arc_rad: float = 0.3,
    max_relative_residual: float = 0.3,
    detrend: bool = True,
    exclude: np.ndarray | None = None,
) -> CenterTrack | None:
    """Sliding estimate of the centre of the IQ arc (M2), for a drifting static component.

    A person moving near the radar changes the fixed reflections, so the
    centre ``C(t)`` of the arc drifts (the DC compensation of B.-K. Park,
    O. Boric-Lubecke and V. M. Lubecke, IEEE T-MTT 55(5), 2007, assumes it
    fixed).  The centre is fitted on sub-windows of *window_s* (2 to 3
    breathing cycles) every *step_s*, with one of the estimators compared by
    Zakrzewski et al. (IEEE Sensors J., 2012): ``"pratt"`` (default),
    ``"geometric"`` (:func:`fit_circle`) or ``"kasa"``.

    Inside a sub-window the centre may already move by as much as the
    breathing arc (a radius or more), and the points then leave the circle:
    measured on 8 mm of breathing at 2.45 GHz, a drift of one radius over
    20 s gives errors of tens of millimetres.  The linear trend of each
    sub-window (complex least-squares slope against time, the breathing
    oscillation averaging out over 2 to 3 cycles) is therefore removed
    before the fit (*detrend*); the error then stays near 1 mm for that
    drift (MOTION_ROBUSTNESS_PLAN.md §3.3).

    The centres are interpolated **linearly** between the middles of the
    valid sub-windows: ``C(t)`` is continuous, so ``z(t) − C(t)`` and its
    angle are continuous too — an update of the centre never creates a phase
    jump.  Invalid sub-windows (short arc, large residual) are skipped.

    Samples marked in *exclude* (in-band anomalies found by M3: a passer-by
    whose echo turns around the centre leaves the circle) are left out of
    the fits; a sub-window with fewer than half of its samples left is
    skipped.

    Parameters
    ----------
    z : numpy.ndarray
        Complex slow-time window, 1-D.
    f_s_hz : float
        Slow-time rate (Hz).
    window_s, step_s : float, optional
        Length and step of the sub-windows (s).
    method : str, optional
        Centre estimator.
    min_arc_rad, max_relative_residual : float, optional
        Validity of a sub-window fit (see :func:`fit_circle`).
    detrend : bool, optional
        Remove the linear drift of each sub-window before fitting it.
    exclude : numpy.ndarray or None, optional
        Boolean mask of the samples left out of the fits, same shape as *z*.

    Returns
    -------
    CenterTrack or None
        ``None`` when no sub-window gives a valid fit.
    """
    fits = {
        "pratt": lambda segment: fit_circle_pratt(segment, min_arc_rad, max_relative_residual),
        "geometric": lambda segment: fit_circle(segment, min_arc_rad=min_arc_rad,
                                                max_relative_residual=max_relative_residual),
        "kasa": lambda segment: fit_circle(segment, n_iterations=0, min_arc_rad=min_arc_rad,
                                           max_relative_residual=max_relative_residual),
    }
    if method not in fits:
        raise ValueError(f"Unknown centre estimator '{method}', use {sorted(fits)}.")
    z = np.asarray(z, dtype=np.complex128)
    n_window = min(z.size, max(4, int(round(window_s * f_s_hz))))
    n_step = max(1, int(round(step_s * f_s_hz)))
    starts = list(range(0, z.size - n_window + 1, n_step))
    if starts[-1] != z.size - n_window:
        starts.append(z.size - n_window)
    centred_time_s = (np.arange(n_window) - (n_window - 1) / 2.0) / f_s_hz
    keep_all = np.ones(z.size, dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)
    circles = []
    for start in starts:
        keep = keep_all[start:start + n_window]
        if keep.sum() < max(4, n_window // 2):
            circles.append(CircleFit(complex(np.nan, np.nan), float("nan"), float("inf"), 0.0, False))
            continue
        segment, segment_t_s = z[start:start + n_window][keep], centred_time_s[keep]
        if detrend:  # centre of the segment at its middle, drift removed
            tc_s = segment_t_s - segment_t_s.mean()
            slope = np.sum(tc_s * (segment - segment.mean())) / np.sum(tc_s ** 2)
            segment = segment - slope * segment_t_s
        circles.append(fits[method](segment))
    times_s = (np.array(starts) + (n_window - 1) / 2.0) / f_s_hz
    centers = np.array([c.center for c in circles])
    valid = np.array([c.valid for c in circles])
    if not valid.any():
        return None
    t_s = np.arange(z.size) / f_s_hz
    per_sample = (np.interp(t_s, times_s[valid], centers[valid].real)
                  + 1j * np.interp(t_s, times_s[valid], centers[valid].imag))
    return CenterTrack(times_s, centers, valid, per_sample)


# ---------------------------------------------------------------------------
# Common rotation (residual local-oscillator offset)
# ---------------------------------------------------------------------------

def remove_common_rotation(z: np.ndarray, f_s_hz: float) -> tuple[np.ndarray, float]:
    """Remove a constant-frequency rotation shared by all the echoes.

    A residual frequency offset ``δf`` between the TX and RX local
    oscillators multiplies the *whole* received signal, static clutter
    included, by ``exp(j2π·δf·t)``.  The DC offset ``C`` then rotates around
    the origin and no fixed circle exists.  The rotation is estimated as the
    least-squares slope of the unwrapped angle of *z* over the window and
    removed.

    Parameters
    ----------
    z : numpy.ndarray
        Complex slow-time samples, 1-D.
    f_s_hz : float
        Sampling rate of *z* (Hz).

    Returns
    -------
    tuple[numpy.ndarray, float]
        ``(derotated samples, estimated δf in Hz)``.

    Notes
    -----
    Assumes that the window holds several breathing periods, so that the
    breathing phase averages out of the linear fit.
    """
    z = np.asarray(z, dtype=np.complex128)
    if z.size < 3:
        return z, 0.0
    t_s = np.arange(z.size) / f_s_hz
    slope_rad_s = np.polyfit(t_s, np.unwrap(np.angle(z)), 1)[0]
    derotated = z * np.exp(-1j * slope_rad_s * t_s)
    return derotated, float(slope_rad_s / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Demodulation
# ---------------------------------------------------------------------------

@dataclass
class DemodulationResult:
    """Output of :func:`demodulate`.

    Attributes
    ----------
    phase_rad : numpy.ndarray
        Demodulated phase (rad), up to a constant.  Arbitrary units (and
        sign) when ``calibrated`` is ``False``.
    method : str
        Demodulation actually used (``"linear"`` after a fallback).
    calibrated : bool
        ``True`` if ``phase_rad`` is a true phase in radians (arctangent or
        DACM on a valid DC compensation).
    circle : CircleFit or None
        Circle used for the DC compensation, if any.
    """

    phase_rad: np.ndarray
    method: str
    calibrated: bool
    circle: CircleFit | None


def arctangent_demodulation(z: np.ndarray) -> np.ndarray:
    """Unwrapped angle (rad) of DC-compensated IQ samples.

    Assumes the DC offset has been removed and that the phase moves by less
    than π between two samples (true for breathing at ≥ 10 Hz slow time).
    """
    return np.unwrap(np.angle(z))


def dacm_demodulation(z: np.ndarray) -> np.ndarray:
    """Extended differentiate-and-cross-multiply (DACM) demodulation (rad).

    ``φ[n] = Σ_{k≤n} (I[k]·ΔQ[k] − Q[k]·ΔI[k]) / (I[k]² + Q[k]²)``

    DACM computes the phase increments directly and needs no unwrapping; it
    assumes small increments between samples (dense slow-time sampling).
    """
    z = np.asarray(z, dtype=np.complex128)
    if z.size == 0:
        return np.zeros(0)
    i, q = z.real, z.imag
    delta_i = np.diff(i, prepend=i[0])
    delta_q = np.diff(q, prepend=q[0])
    power = np.maximum(i * i + q * q, 1e-30)
    phase_increments = (i * delta_q - q * delta_i) / power
    return np.cumsum(phase_increments)


def linear_demodulation(z: np.ndarray) -> np.ndarray:
    """Small-angle (linear) demodulation by projection on the principal axis.

    For a small phase excursion the arc degenerates into a segment; the
    projection of the mean-removed samples on their principal axis (PCA) is
    proportional to ``φ``.  Scale and sign are unknown, so the output is
    normalised to a unit standard deviation.  Valid only if ``4π·Δd/λ ≪ 1``.
    """
    z = np.asarray(z, dtype=np.complex128)
    centred = z - z.mean()
    covariance = np.cov(np.vstack((centred.real, centred.imag)))
    _, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, -1]
    projection = centred.real * principal_axis[0] + centred.imag * principal_axis[1]
    std = projection.std()
    return projection / std if std > 0 else projection


def demodulate(
    z: np.ndarray,
    method: str = "arctan",
    dc_method: str = "circle",
    min_arc_rad: float = 0.3,
    max_relative_residual: float = 0.3,
) -> DemodulationResult:
    """DC-compensate and demodulate a window of slow-time IQ samples.

    Parameters
    ----------
    z : numpy.ndarray
        Complex slow-time samples of the analysis window, 1-D.
    method : str, optional
        ``"arctan"`` (default), ``"dacm"`` or ``"linear"``.
    dc_method : str, optional
        ``"circle"`` (circle fitting, recommended), ``"mean"`` (subtract the
        window mean — correct only if the samples cover a full circle) or
        ``"none"`` (samples already centred).
    min_arc_rad, max_relative_residual : float, optional
        Validity limits of the circle fit (see :func:`fit_circle`).

    Returns
    -------
    DemodulationResult
        When the circle fit is rejected, falls back to
        :func:`linear_demodulation` and says so (``method == "linear"``).
    """
    if method not in VALID_DEMODULATIONS:
        raise ValueError(f"Unknown demodulation '{method}', use {VALID_DEMODULATIONS}.")
    if dc_method not in VALID_DC_METHODS:
        raise ValueError(f"Unknown DC method '{dc_method}', use {VALID_DC_METHODS}.")
    z = np.asarray(z, dtype=np.complex128)

    if method == "linear":
        return DemodulationResult(linear_demodulation(z), "linear", False, None)

    circle: CircleFit | None = None
    if dc_method == "circle":
        circle = fit_circle(z, min_arc_rad=min_arc_rad, max_relative_residual=max_relative_residual)
        if not circle.valid:
            logger.debug(
                "Circle fit rejected (arc=%.2f rad, residual=%.2f) — linear demodulation",
                circle.arc_span_rad, circle.rms_residual,
            )
            return DemodulationResult(linear_demodulation(z), "linear", False, circle)
        centred = z - circle.center
    elif dc_method == "mean":
        centred = z - z.mean()
    else:
        centred = z

    if method == "arctan":
        phase_rad = arctangent_demodulation(centred)
    else:
        phase_rad = dacm_demodulation(centred)
    return DemodulationResult(phase_rad, method, True, circle)


def demodulate_tracked(
    z: np.ndarray,
    f_s_hz: float,
    method: str = "arctan",
    window_s: float = 10.0,
    step_s: float = 1.0,
    estimator: str = "pratt",
    min_arc_rad: float = 0.3,
    max_relative_residual: float = 0.3,
    exclude: np.ndarray | None = None,
) -> DemodulationResult:
    """Phase demodulation around a drifting centre (M2, MOTION_ROBUSTNESS_PLAN.md §3.3).

    The centre is tracked by :func:`track_center` and subtracted sample by
    sample before the arctangent (or DACM): ``φ(t) = unwrap(arg(z(t) − C(t)))``
    (Park et al., IEEE T-MTT 2007, with a time-varying DC offset).  Without
    any valid sub-window, the window is demodulated as by :func:`demodulate`
    with a single circle fit.  *exclude* (samples left out of the centre
    fits) is passed to :func:`track_center`.

    Returns
    -------
    DemodulationResult
        Method ``"arctan_tracked"`` (or ``"dacm_tracked"``); its ``circle``
        summarises the end of the window (latest centre, median radius), for
        display.
    """
    z = np.asarray(z, dtype=np.complex128)
    track = track_center(z, f_s_hz, window_s, step_s, estimator, min_arc_rad, max_relative_residual,
                         exclude=exclude)
    if track is None:
        return demodulate(z, method, "circle", min_arc_rad, max_relative_residual)
    centred = z - track.center_per_sample
    phase_rad = arctangent_demodulation(centred) if method == "arctan" else dacm_demodulation(centred)
    radius = float(np.median(np.abs(centred)))
    residual = float(np.sqrt(np.mean((np.abs(centred) - radius) ** 2)) / radius) if radius > 0 else np.inf
    summary = CircleFit(complex(track.center_per_sample[-1]), radius, residual,
                        _arc_span_rad(np.angle(centred)), True)
    return DemodulationResult(phase_rad, f"{method}_tracked", True, summary)


def phase_to_displacement_m(phase_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Convert a two-way phase (rad) into a radial displacement (m).

    ``d = −λ / (4π) · φ``: the minus sign follows the convention
    ``x ∝ exp(−j·4π·R/λ)`` — an increase of the range decreases the phase.
    """
    return -float(wavelength_m) / (4.0 * np.pi) * np.asarray(phase_rad, dtype=np.float64)
