"""Cheap smoke-run diagnostics and ranking for AOI candidates.

Smoke scores are deliberately advisory. They can prune only egregious failures
using the existing debug gates; otherwise they rank batches before expensive full
runs. Final selection continues to use full-run and averaged metrics.
"""

from __future__ import annotations

from typing import Any

from mle_star_agent import config
from mle_star_agent.shared import metric_guard
from mle_star_agent.shared.diagnosis_scorer import (
    detect_early_collapse,
    parse_calibration_stats,
    parse_epoch_logs,
    parse_threshold_curve,
)
from mle_star_agent.shared.metrics_parser import (
    metrics_to_dict,
    parse_metrics,
    parse_probe_metrics,
)


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def _metric(metrics: dict, key: str, default: float = 0.0) -> float:
    return _clamp(metrics.get(key, default))


def _prob_gap_score(value: Any) -> float:
    # A gap around 0.25 is already useful for triage; larger gaps saturate.
    try:
        gap = float(value)
    except (TypeError, ValueError):
        return 0.0
    return _clamp(gap / 0.25)


def _threshold_score(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.05 <= threshold <= 0.95:
        return 1.0
    return 0.25


def _loss_trend_score(epoch_logs: list[dict]) -> float:
    if len(epoch_logs) < 2:
        return 0.0

    def loss(row: dict) -> float | None:
        for key in ("val_loss", "loss", "train_loss"):
            if row.get(key) is not None:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    return None
        return None

    first = loss(epoch_logs[0])
    last = loss(epoch_logs[-1])
    if first is None or last is None or first <= 0:
        return 0.0
    return _clamp((first - last) / first)


def score_smoke_metrics(metrics: dict | None, diagnostics: dict | None = None) -> float | None:
    """Return an advisory 0-1 smoke score, or None when metrics are unavailable."""
    if not metrics:
        return None

    diagnostics = diagnostics or {}
    score = (
        0.35 * _metric(metrics, "ng_recall")
        + 0.20 * (1.0 - _metric(metrics, "miss_rate", 1.0))
        + 0.18 * (1.0 - _metric(metrics, "overkill_rate", 1.0))
        + 0.10 * _metric(metrics, "accuracy")
        + 0.07 * _metric(metrics, "f1")
        + 0.05 * _metric(metrics, "roc_auc")
        + 0.03 * _prob_gap_score(metrics.get("prob_gap"))
        + 0.02 * _threshold_score(metrics.get("threshold"))
    )

    epoch_logs = diagnostics.get("epoch_logs") or []
    if epoch_logs:
        score += 0.03 * _loss_trend_score(epoch_logs)
    early_collapse = diagnostics.get("early_collapse") or {}
    if early_collapse.get("detected"):
        score -= 0.15

    return round(_clamp(score), 6)


def is_egregious_smoke_metrics(metrics: dict | None) -> bool:
    """Only the existing loose debug gates may prune from smoke metrics."""
    if not metrics:
        return False
    return (
        float(metrics.get("overkill_rate", 0.0) or 0.0) > config.DEBUG_PREDICT_OVERKILL_MAX
        or float(metrics.get("ng_recall", 1.0) or 1.0) < config.DEBUG_PREDICT_NG_RECALL_MIN
    )


def build_smoke_diagnostics(stdout: str, duration_ms: float, *, context: str) -> dict:
    """Parse and score all cheap smoke signals available in stdout."""
    parsed = parse_metrics(stdout)
    guarded = metric_guard.guard_metrics(
        parsed,
        duration_ms,
        mode="subsample",
        context=context,
    ) if parsed is not None else None
    metrics = metrics_to_dict(guarded) if guarded else None
    epoch_logs = parse_epoch_logs(stdout)
    early_collapse = detect_early_collapse(epoch_logs) if epoch_logs else None
    diagnostics = {
        "metrics": metrics,
        "score": None,
        "pruned": False,
        "prune_reason": None,
        "probe_metrics": parse_probe_metrics(stdout),
        "calibration_stats": parse_calibration_stats(stdout),
        "threshold_curve": parse_threshold_curve(stdout),
        "epoch_logs": epoch_logs,
        "early_collapse": early_collapse,
    }
    diagnostics["score"] = score_smoke_metrics(metrics, diagnostics)
    if is_egregious_smoke_metrics(metrics):
        diagnostics["pruned"] = True
        diagnostics["prune_reason"] = "smoke_pruned_egregious"
    return diagnostics


def select_full_run_slots(results: list[dict]) -> dict[int, str]:
    """Choose Phase 1 candidates for full runs from smoke diagnostics.

    Missing smoke metrics are always selected. Metric-bearing candidates are
    ranked by smoke score; top K proceed, plus candidates within the uncertainty
    band of the Kth score.
    """
    selected: dict[int, str] = {}
    scored: list[dict] = []

    for result in results:
        if result.get("status") != "smoke_pending_full":
            continue
        slot = result.get("slot")
        if slot is None:
            continue
        if result.get("smoke_metrics") is None or result.get("smoke_score") is None:
            selected[int(slot)] = "missing_smoke_metrics"
        else:
            scored.append(result)

    scored.sort(key=lambda r: float(r.get("smoke_score", 0.0) or 0.0), reverse=True)
    top_k = max(0, int(getattr(config, "PHASE1_SMOKE_TOP_K", 2)))
    band = max(0.0, float(getattr(config, "PHASE1_SMOKE_UNCERTAINTY_BAND", 0.05)))
    if top_k <= 0 or not scored:
        return selected

    cutoff_index = min(top_k, len(scored)) - 1
    cutoff = float(scored[cutoff_index].get("smoke_score", 0.0) or 0.0)
    for index, result in enumerate(scored):
        slot = int(result["slot"])
        score = float(result.get("smoke_score", 0.0) or 0.0)
        if index < top_k:
            selected[slot] = "top_smoke_rank"
        elif score >= cutoff - band:
            selected[slot] = "smoke_uncertainty_band"

    return selected
