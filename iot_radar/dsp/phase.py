"""DC compensation and phase demodulation of a slow-time IQ window.

Place in the chain: after the decimation to slow time, before the
displacement analysis.  The slow-time signal is::

    x[n] = C + A · exp(j(θ0 − 4π d[n] / λ)) + w[n]

so the samples lie on an arc of radius ``A`` centred on the DC offset ``C``
(leakage, static clutter, receiver DC).  Hence: fit the circle and subtract
``C``, take the unwrapped angle, convert with ``d = −λ/(4π) · φ``.

.. warning::
   A high-pass filter is not a substitute for subtracting ``C``: it also
   removes the mean of ``A·exp(jφ)``, and the angle is then wrong.

Reference: G. Paterniani *et al.*, "Radar-based monitoring of vital signs: a
tutorial overview", Proc. IEEE 111(3), 2023.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Circle fitting (DC offset compensation)
# ---------------------------------------------------------------------------

@dataclass
class CircleFit:
    """Fitted circle: centre ``C`` and radius ``A`` (IQ units), RMS residual / radius,
    arc span around the centre (rad), and whether the fit is reliable."""

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
    """Fit a circle to complex samples: algebraic start, then Gauss–Newton.

    Start: I. Kåsa, "A circle fitting procedure and its error analysis", IEEE
    Trans. Instrum. Meas. IM-25(1), 1976 (linear least squares on
    ``x² + y² + D·x + E·y + F = 0``); refinement on the distance ``|z − c| − r``.

    The fit is invalid below *min_arc_rad* (short arc: centre ill-determined)
    or above *max_relative_residual* (a noise cloud gives ≈ 0.4; 0.3 ≈ an arc
    SNR of 10 dB).  The caller then uses :func:`linear_demodulation`.
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


# ---------------------------------------------------------------------------
# Common rotation (residual local-oscillator offset)
# ---------------------------------------------------------------------------

def remove_common_rotation(z: np.ndarray, f_s_hz: float) -> tuple[np.ndarray, float]:
    """Remove a rotation ``exp(j2π·δf·t)`` shared by all echoes; return ``(z, δf in Hz)``.

    A TX/RX local-oscillator offset makes the whole signal, ``C`` included,
    rotate, so no fixed circle exists.  ``δf`` is the slope of the unwrapped
    angle over the window (assumes several breaths per window).
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

    ``phase_rad`` is in radians (up to a constant) only when ``calibrated``
    (arctangent on a valid circle); ``method`` is ``"arctan"`` or ``"linear"``.
    """

    phase_rad: np.ndarray
    method: str
    calibrated: bool
    circle: CircleFit | None


def arctangent_demodulation(z: np.ndarray) -> np.ndarray:
    """Unwrapped angle (rad) of DC-compensated samples (phase step < π per sample)."""
    return np.unwrap(np.angle(z))


def linear_demodulation(z: np.ndarray) -> np.ndarray:
    """Small-angle demodulation: projection on the principal axis (PCA), unit std.

    For a small excursion the arc is almost a segment; the projection is
    proportional to ``φ``, with unknown scale and sign.
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
    min_arc_rad: float = 0.3,
    max_relative_residual: float = 0.3,
) -> DemodulationResult:
    """Circle fit + arctangent around its centre; linear demodulation if the fit is rejected."""
    z = np.asarray(z, dtype=np.complex128)
    circle = fit_circle(z, min_arc_rad=min_arc_rad, max_relative_residual=max_relative_residual)
    if not circle.valid:
        logger.debug(
            "Circle fit rejected (arc=%.2f rad, residual=%.2f) — linear demodulation",
            circle.arc_span_rad, circle.rms_residual,
        )
        return DemodulationResult(linear_demodulation(z), "linear", False, circle)
    return DemodulationResult(arctangent_demodulation(z - circle.center), "arctan", True, circle)


def phase_to_displacement_m(phase_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Two-way phase (rad) → radial displacement (m): ``d = −λ / (4π) · φ``."""
    return -float(wavelength_m) / (4.0 * np.pi) * np.asarray(phase_rad, dtype=np.float64)
