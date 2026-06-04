"""Saturating power-law learning-curve extrapolation (no scipy).

Given a list of per-epoch metric values (e.g. ``val_ng_recall`` over the debug
micro-run's short curve) this fits ``y = a + b * t^(-c)`` and projects the
saturated final value ``a``. It is used by the phase-2 evaluator to prune doomed
candidates at the debug -> full boundary, so it is intentionally PURE and NEVER
raises: any bad / degenerate / too-short input returns ``(None, 0.0)`` and the
caller simply falls through to the full run.

Fit method (numpy log-linear least squares, no scipy):

    y = a + b * t^(-c)  saturates to the asymptote ``a`` as ``t -> inf``.
    For a *fixed* candidate asymptote ``a`` we have |y - a| = |b| * t^(-c), so

        log|y - a| = log|b| - c * log(t)

    is LINEAR in ``log(t)``. We grid-search the asymptote ``a`` (above the data
    for a rising curve, below it for a falling curve), fit that line by ordinary
    least squares for each candidate, and keep the asymptote whose log-log fit has
    the best R^2 *and* a valid saturating decay (slope < 0, i.e. c = -slope > 0).
    ``projected_final`` is the best ``a``; ``fit_quality`` is that R^2 clamped to
    [0, 1].
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from mle_star_agent import config


def _clean_series(points: Sequence) -> Optional[np.ndarray]:
    """Coerce ``points`` to a 1-D finite float array, or None if anything is off."""
    try:
        vals = []
        for p in points:
            f = float(p)
            if not np.isfinite(f):
                return None
            vals.append(f)
        return np.asarray(vals, dtype=float)
    except (TypeError, ValueError):
        return None


def _loglinear_fit(log_t: np.ndarray, log_m: np.ndarray) -> Tuple[float, float]:
    """Least-squares fit ``log_m ~ slope * log_t + intercept``.

    Returns ``(slope, r2)`` where r2 is the coefficient of determination.
    """
    slope, intercept = np.polyfit(log_t, log_m, 1)
    pred = slope * log_t + intercept
    ss_res = float(np.sum((log_m - pred) ** 2))
    ss_tot = float(np.sum((log_m - float(np.mean(log_m))) ** 2))
    if ss_tot <= 1e-12:
        return float(slope), 0.0
    return float(slope), 1.0 - ss_res / ss_tot


def project_power_law(
    points: Sequence[float],
    min_epochs: Optional[int] = None,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[Optional[float], float]:
    """Fit a saturating power-law to ``points`` (metric per epoch, t = 1..n).

    Args:
        points: per-epoch metric values, oldest first.
        min_epochs: minimum number of points required; defaults to
            ``config.CURVE_ABORT_MIN_EPOCHS``.
        value_range: valid ``(lo, hi)`` range for the metric. Power-law asymptote
            fitting from a few points is ill-posed — a slowly-saturating curve fits
            equally well for many asymptotes — so the projection is clamped to this
            range. AOI per-epoch metrics (ng_recall, overkill) are rates in [0, 1].

    Returns:
        ``(projected_final, fit_quality)`` where ``projected_final`` is the
        saturated asymptote (clamped to ``value_range``) and ``fit_quality`` is an
        R^2 in [0, 1]. Returns ``(None, 0.0)`` for too few points or a
        poor/degenerate fit. Never raises.
    """
    try:
        if min_epochs is None:
            min_epochs = config.CURVE_ABORT_MIN_EPOCHS

        y = _clean_series(points)
        if y is None or len(y) < int(min_epochs):
            return (None, 0.0)

        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        log_t = np.log(t)

        rng = float(y.max() - y.min())
        if rng <= 1e-9:
            # Flat curve: no curvature to recover a decay exponent from -> degenerate.
            return (None, 0.0)

        # Place the asymptote strictly beyond the data so |y - a| > 0 everywhere.
        # A rising curve approaches from below (a > max y); a falling curve from
        # above (a < min y). Both are tried and the best-fitting one wins.
        offsets = rng * np.concatenate((
            np.linspace(0.02, 1.0, 25),
            np.linspace(1.2, 4.0, 15),
        ))

        best_a: Optional[float] = None
        best_r2 = -np.inf
        y_max = float(y.max())
        y_min = float(y.min())

        for off in offsets:
            for a in (y_max + off, y_min - off):  # rising, then falling
                m = np.abs(y - a)
                if np.any(m <= 1e-12):
                    continue
                slope, r2 = _loglinear_fit(log_t, np.log(m))
                # A saturating power-law needs the gap to DECAY with t: slope < 0
                # (slope = -c, so c = -slope > 0). Reject anything else.
                if slope >= 0:
                    continue
                if r2 > best_r2:
                    best_r2 = r2
                    best_a = float(a)

        if best_a is None:
            return (None, 0.0)

        lo, hi = float(value_range[0]), float(value_range[1])
        projected = float(max(lo, min(hi, best_a)))
        fit_quality = float(max(0.0, min(1.0, best_r2)))
        return (projected, fit_quality)
    except Exception:
        return (None, 0.0)
