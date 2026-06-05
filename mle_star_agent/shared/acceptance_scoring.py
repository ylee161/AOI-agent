from __future__ import annotations

from typing import Any, Mapping

from mle_star_agent import config


def _metric(metrics: Any, name: str, default: float) -> float:
    if metrics is None:
        return default
    if isinstance(metrics, Mapping):
        cv_aliases = {
            "accuracy": "mean_val_accuracy",
            "ng_recall": "worst_fold_val_ng_recall",
            "miss_rate": "worst_fold_val_miss_rate",
            "overkill_rate": "mean_val_overkill",
        }
        if name in cv_aliases and cv_aliases[name] in metrics:
            value = metrics.get(cv_aliases[name], default)
        else:
            value = metrics.get(name, default)
    else:
        value = getattr(metrics, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metrics_view(metrics: Any) -> dict:
    return {
        "accuracy": _metric(metrics, "accuracy", 0.0),
        "ng_recall": _metric(metrics, "ng_recall", 0.0),
        "miss_rate": _metric(metrics, "miss_rate", 1.0),
        "overkill_rate": _metric(metrics, "overkill_rate", 1.0),
        "f1": _metric(metrics, "f1", 0.0),
        "roc_auc": _metric(metrics, "roc_auc", 0.0),
        "prob_gap": _metric(metrics, "prob_gap", 0.0),
    }


def passes_relaxed_acceptance(metrics: Any) -> bool:
    m = metrics_view(metrics)
    return (
        m["accuracy"] >= config.ACCURACY_RELAXED_MIN
        and m["ng_recall"] >= config.NG_RECALL_RELAXED_MIN
        and m["miss_rate"] <= config.MISS_RATE_RELAXED_MAX
        and m["overkill_rate"] <= config.OVERKILL_RELAXED_MAX
    )


def passes_final_acceptance(metrics: Any) -> bool:
    m = metrics_view(metrics)
    return (
        m["accuracy"] >= config.ACCURACY_FINAL_MIN
        and m["ng_recall"] >= config.NG_RECALL_FINAL_MIN
        and m["miss_rate"] <= config.MISS_RATE_FINAL_MAX
        and m["overkill_rate"] <= config.OVERKILL_FINAL_MAX
    )


def acceptance_distance(metrics: Any) -> float:
    """
    Lower is better. Each gap is normalized by its relaxed budget, then weighted
    by industrial priority (P0×4, P1×3, P2×2, P4×1) so the scalar reflects the
    spec's P0→P1→P2→P4 order. Used for ablation impact scoring and as a tiebreaker
    when both candidates have the same relaxed-acceptance status.
    """
    m = metrics_view(metrics)
    miss_gap = max(0.0, m["miss_rate"] - config.MISS_RATE_RELAXED_MAX)
    recall_gap = max(0.0, config.NG_RECALL_RELAXED_MIN - m["ng_recall"])
    overkill_gap = max(0.0, m["overkill_rate"] - config.OVERKILL_RELAXED_MAX)
    accuracy_gap = max(0.0, config.ACCURACY_RELAXED_MIN - m["accuracy"])

    return (
        4.0 * miss_gap / max(config.MISS_RATE_RELAXED_MAX, 1e-9)
        + 3.0 * recall_gap / max(config.NG_RECALL_RELAXED_MIN, 1e-9)
        + 2.0 * overkill_gap / max(config.OVERKILL_RELAXED_MAX, 1e-9)
        + 1.0 * accuracy_gap / max(config.ACCURACY_RELAXED_MIN, 1e-9)
    )


def is_acceptance_improvement(new_metrics: Any, current_metrics: Any) -> bool:
    """Decide whether `new_metrics` should replace `current_metrics`.

    Policy (revised — Mini-goal 5, Fix C):

    1. Relaxed-acceptance status dominates. A candidate that meets the relaxed
       §9.1 operating-point budget always beats one that does not, and vice
       versa.
    2. When both share the same relaxed-acceptance status (both pass OR both
       fail), optimize the weighted `acceptance_distance` jointly. Lower is
       better.  miss_rate is still weighted highest inside that distance
       (P0, 4x / relaxed-budget), so defect escape remains strongly penalised —
       but a move that raises miss_rate *slightly* while cutting overkill enough
       to lower the joint distance is now ACCEPTED.

    This replaces the previous hard P0 anti-regression floor
    (`new.miss_rate > current.miss_rate + 1e-6 -> reject`). That floor pinned
    the loop to a failing operating point: the current best miss_rate (~0.0645)
    already exceeds MISS_RATE_RELAXED_MAX (0.03), so every overkill-reducing move
    nudged miss_rate up and was auto-rejected, and `no_improve_count` could only
    climb. Optimizing the weighted distance lets the loop trade a small miss
    regression for a large overkill reduction when that genuinely improves the
    joint objective.
    """
    new_pass = passes_relaxed_acceptance(new_metrics)
    current_pass = passes_relaxed_acceptance(current_metrics)

    # Views computed up-front so they are in scope for the tie-breaker path.
    new = metrics_view(new_metrics)
    current = metrics_view(current_metrics)

    # 1. Relaxed-acceptance status leads.
    if new_pass and not current_pass:
        return True
    if current_pass and not new_pass:
        return False

    # 2. Same relaxed status (both pass OR both fail): joint weighted distance.
    #    No hard miss_rate floor — miss_rate's P0 weight inside acceptance_distance
    #    does the prioritisation without blocking overkill-reducing moves.
    new_distance = acceptance_distance(new_metrics)
    current_distance = acceptance_distance(current_metrics)
    if new_distance < current_distance:
        return True
    if new_distance > current_distance:
        return False

    # 3. Distances tie: lexicographic P0->P1->P2->P4 tie-breaker.
    #    Order must follow the spec priority: miss_rate (P0), then ng_recall (P1),
    #    then overkill_rate (P2), then accuracy (P4). f1 is a final tiebreaker.
    return (
        new["miss_rate"],
        -new["ng_recall"],
        new["overkill_rate"],
        -new["accuracy"],
        -new["f1"],
    ) < (
        current["miss_rate"],
        -current["ng_recall"],
        current["overkill_rate"],
        -current["accuracy"],
        -current["f1"],
    )
