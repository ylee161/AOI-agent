"""Degenerate-metric guard for the metric persistence boundary.

This module exists because a single degenerate probe poisoned the entire
decision stream once before: `training_schedule` (ablation_0_variant_8)
ran for 2.6 s on an 8-row DUMMY split (1 G + 1 NG, identical scores) and
emitted roc_auc=0.0, which then flowed into candidate_scores.json, the
ablation summaries, and diagnosis_0.json — driving the whole loop toward a
non-existent "low capacity" failure mode.

The guard is called at every place a parsed-metrics record is about to be
written to a checkpoint artifact used for decisions. If the run looks
degenerate it is rejected: the caller drops the metrics (treats the run as
failed) so the degenerate numbers never enter candidate_scores.json, ablation
summaries, diagnosis inputs, or any other persisted decision artifact.

Thresholds live here (not in config.py) so they are self-documenting and so
this safety floor cannot be silently relaxed by tuning the acceptance budget.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# --- rejection thresholds (a run failing ANY of these is degenerate) ---------
MIN_RUNTIME_SECONDS = 30.0   # a real CPU ResNet18 eval never finishes this fast
MIN_NG_COUNT = 5             # too few NG to estimate miss_rate / AUC honestly
MIN_G_COUNT = 5              # too few G to estimate overkill / AUC honestly
PROB_GAP_EPS = 1e-9          # prob_gap at/below this == no separability signal
SCORE_IDENTICAL_STD_EPS = 1e-6  # std of predicted scores below this == flat

# Verification modes that are allowed to run fast / on a subsample. These only
# waive the RUNTIME check — a run that is degenerate by separability
# (prob_gap==0, flat scores) is rejected even in verification mode.
_VERIFICATION_MODES = {"dry_run", "smoke", "subsample", "verify"}


class DegenerateMetricsError(ValueError):
    """Raised when degenerate metrics are submitted for persistence."""


def _get(metrics: Any, name: str, default: float = 0.0) -> float:
    if metrics is None:
        return default
    if isinstance(metrics, Mapping):
        value = metrics.get(name, default)
    else:
        value = getattr(metrics, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_verification_mode(mode: Optional[str]) -> bool:
    if mode and str(mode).strip().lower() in _VERIFICATION_MODES:
        return True
    # Also honour environment switches so a manually launched smoke run is exempt
    # from the runtime floor without code changes at each call site.
    for var in ("AOI_DRY_RUN", "AOI_SMOKE", "AOI_SUBSAMPLE", "AOI_VERIFY"):
        v = os.environ.get(var, "").strip().lower()
        if v and v not in ("0", "false", "no", ""):
            return True
    return False


def scores_effectively_identical(scores: Optional[Sequence[float]]) -> bool:
    """True when every predicted score is the same (degenerate AUC)."""
    if not scores:
        return False
    clean = [float(s) for s in scores if s is not None]
    if len(clean) < 2:
        return False
    lo, hi = min(clean), max(clean)
    return (hi - lo) <= SCORE_IDENTICAL_STD_EPS


def degenerate_rejection_reason(
    metrics: Any,
    duration_ms: Optional[float] = None,
    *,
    mode: Optional[str] = None,
    scores: Optional[Sequence[float]] = None,
) -> Optional[str]:
    """Return a human-readable reason if these metrics are degenerate, else None.

    `metrics` may be an AOIMetrics, a metrics dict, or None. `duration_ms` is the
    wall-clock runtime of the run that produced the metrics. `scores` is the
    optional list of per-sample NG probabilities (used for the flat-scores check).
    """
    if metrics is None:
        return "no metrics parsed from run output"

    verification = _is_verification_mode(mode)

    if duration_ms is not None and not verification:
        runtime_s = float(duration_ms) / 1000.0
        if runtime_s < MIN_RUNTIME_SECONDS:
            return (
                f"runtime {runtime_s:.1f}s < {MIN_RUNTIME_SECONDS:.0f}s floor "
                f"(implausibly fast — likely ran on a dummy/empty split)"
            )

    ng_count = int(_get(metrics, "ng_count", 0))
    g_count = int(_get(metrics, "g_count", 0))
    if ng_count < MIN_NG_COUNT:
        return f"ng_count {ng_count} < {MIN_NG_COUNT} (too few NG to score honestly)"
    if g_count < MIN_G_COUNT:
        return f"g_count {g_count} < {MIN_G_COUNT} (too few G to score honestly)"

    prob_gap = _get(metrics, "prob_gap", default=float("nan"))
    # prob_gap is the NG-mean minus G-mean separability margin. Exactly-zero (or
    # negative-zero / sub-eps) means the model produced no separation at all.
    if prob_gap == prob_gap and abs(prob_gap) <= PROB_GAP_EPS:  # not NaN and ~0
        return f"prob_gap {prob_gap} == 0 (no separability — degenerate scores)"

    if scores_effectively_identical(scores):
        return "all predicted scores effectively identical (degenerate AUC)"

    return None


def is_persistable(
    metrics: Any,
    duration_ms: Optional[float] = None,
    *,
    mode: Optional[str] = None,
    scores: Optional[Sequence[float]] = None,
) -> bool:
    """Convenience boolean wrapper around :func:`degenerate_rejection_reason`."""
    return degenerate_rejection_reason(
        metrics, duration_ms, mode=mode, scores=scores
    ) is None


def guard_metrics(
    metrics: Any,
    duration_ms: Optional[float] = None,
    *,
    context: str = "",
    mode: Optional[str] = None,
    scores: Optional[Sequence[float]] = None,
    raise_on_reject: bool = False,
) -> Any:
    """Gate a metrics record at a persistence boundary.

    Returns the metrics unchanged when they are acceptable. When degenerate, logs
    a clear error and either returns ``None`` (so the caller persists the run as a
    *failed* candidate — the metrics never enter the decision artifacts) or raises
    :class:`DegenerateMetricsError` when ``raise_on_reject`` is True.
    """
    reason = degenerate_rejection_reason(
        metrics, duration_ms, mode=mode, scores=scores
    )
    if reason is None:
        return metrics

    msg = (
        f"REJECTED degenerate metrics{f' [{context}]' if context else ''}: "
        f"{reason}. Metrics will NOT be persisted as a valid candidate "
        f"(guarding against dummy-split poisoning)."
    )
    logger.error(msg)
    if raise_on_reject:
        raise DegenerateMetricsError(msg)
    return None
