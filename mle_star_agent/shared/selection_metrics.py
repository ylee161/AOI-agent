from __future__ import annotations

from functools import cmp_to_key
from typing import Any, Callable, Iterable, Mapping

from mle_star_agent.shared.acceptance_scoring import is_acceptance_improvement, metrics_view


AVERAGED_EVALUATION_KEY = "selection_evaluation"

METRIC_AVERAGE_KEYS = (
    "accuracy",
    "ng_recall",
    "miss_rate",
    "overkill_rate",
    "f1",
    "avg_latency_ms",
    "threshold",
    "ng_count",
    "g_count",
    "tp",
    "tn",
    "fp",
    "fn",
    "roc_auc",
    "prob_gap",
)


def average_metrics_dicts(metric_dicts: Iterable[Mapping[str, Any]]) -> dict:
    metric_dicts = [dict(m) for m in metric_dicts]
    averaged = {}
    for key in METRIC_AVERAGE_KEYS:
        values = [float(m.get(key, 0.0) or 0.0) for m in metric_dicts]
        averaged[key] = sum(values) / len(values) if values else 0.0

    for key in ("ng_count", "g_count", "tp", "tn", "fp", "fn"):
        averaged[key] = int(round(averaged[key]))
    return averaged


def build_selection_evaluation(
    *,
    seeds: Iterable[int],
    seed_results: list[dict],
    averaged_metrics: Mapping[str, Any] | None = None,
) -> dict:
    expected_seeds = tuple(seeds)
    if averaged_metrics is None:
        successful = [r for r in seed_results if r.get("metrics") is not None]
        status = "incomplete" if successful else "failed"
        return {
            "status": status,
            "seeds": list(expected_seeds),
            "metrics": None,
            "per_seed": seed_results,
            "successful_seed_count": len(successful),
            "expected_seed_count": len(expected_seeds),
            "failure_reason": "one_or_more_seed_runs_failed",
        }

    return {
        "status": "success",
        "seeds": list(expected_seeds),
        "metrics": dict(averaged_metrics),
        "per_seed": seed_results,
        "successful_seed_count": len(expected_seeds),
        "expected_seed_count": len(expected_seeds),
    }


def selection_metrics_for_record(record: Mapping[str, Any]) -> dict:
    selection_eval = record.get(AVERAGED_EVALUATION_KEY)
    if isinstance(selection_eval, Mapping) and selection_eval.get("status") == "success":
        averaged = selection_eval.get("metrics")
        if isinstance(averaged, Mapping):
            return dict(averaged)
    cv_eval = record.get("cv_evaluation")
    if isinstance(cv_eval, Mapping) and cv_eval.get("status") == "success":
        metrics = cv_eval.get("metrics")
        if isinstance(metrics, Mapping):
            return dict(metrics)
    metrics = record.get("metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def select_best_record(
    records: Iterable[Mapping[str, Any]],
    *,
    metric_getter: Callable[[Mapping[str, Any]], dict] = selection_metrics_for_record,
) -> Mapping[str, Any] | None:
    records = [r for r in records if metric_getter(r)]
    if not records:
        return None

    def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
        left_metrics = metric_getter(left)
        right_metrics = metric_getter(right)
        if is_acceptance_improvement(left_metrics, right_metrics):
            return -1
        if is_acceptance_improvement(right_metrics, left_metrics):
            return 1

        left_view = metrics_view(left_metrics)
        right_view = metrics_view(right_metrics)
        left_tie = (
            left_view["miss_rate"],
            -left_view["ng_recall"],
            left_view["overkill_rate"],
            -left_view["accuracy"],
            -left_view["f1"],
        )
        right_tie = (
            right_view["miss_rate"],
            -right_view["ng_recall"],
            right_view["overkill_rate"],
            -right_view["accuracy"],
            -right_view["f1"],
        )
        return (left_tie > right_tie) - (left_tie < right_tie)

    return sorted(records, key=cmp_to_key(compare))[0]
