"""Deterministic analytical-state labelling for the Phase 2 refinement loop.

The refinement inner loop chases a TWO-TIER, prioritised acceptance budget
(config.py §9.1 relaxed / §9.2 final). Across deep inner-loop iterations the LLM
can lose track of WHICH target is currently the binding constraint. This module
derives a single, deterministic label from metrics already in state — no LLM call
invents it — mirroring the project's pre-LLM diagnosis-scoring design.

The label is computed by a fixed priority ladder evaluated against the RELAXED
tier until relaxed acceptance is met, after which the only distinction is whether
the FINAL tier is also met. P0 miss_rate ALWAYS outranks P2 overkill.

This module is read-only: it never mutates state and never raises.
"""

from __future__ import annotations

from typing import Any, Mapping

from mle_star_agent import config
from mle_star_agent.shared.acceptance_scoring import (
    passes_final_acceptance,
    passes_relaxed_acceptance,
)

_DEFAULT = {
    "label": "OPTIMIZING_P0_MISS_RATE",
    "detail": "no metrics available yet — defaulting to P0 miss-rate focus",
}


def _read_float(state: Mapping, key: str):
    """Return state[key] as a float, or None if absent/None/unparseable."""
    try:
        value = state.get(key)
    except Exception:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_analytical_state(state: Any) -> dict:
    """Derive {"label", "detail"} for the active acceptance target from `state`.

    Reads current-best metrics (current_best_score == ng_recall, best_miss_rate,
    best_overkill_rate, best_accuracy) with the evaluator/ideator convention that
    miss_rate falls back to max(0, 1 - ng_recall) when absent. Never raises; when
    metrics are entirely absent it defaults to OPTIMIZING_P0_MISS_RATE.
    """
    try:
        if not isinstance(state, Mapping):
            return dict(_DEFAULT)

        raw_ng = _read_float(state, "current_best_score")
        raw_miss = _read_float(state, "best_miss_rate")
        raw_overkill = _read_float(state, "best_overkill_rate")
        raw_accuracy = _read_float(state, "best_accuracy")

        # Entirely absent -> P0 default (matches the "no metrics yet" first pass).
        if (
            raw_ng is None
            and raw_miss is None
            and raw_overkill is None
            and raw_accuracy is None
        ):
            return dict(_DEFAULT)

        # Worst-case defaults mirror acceptance_scoring.metrics_view so a missing
        # metric reads as "not yet met" rather than silently passing.
        ng_recall = raw_ng if raw_ng is not None else 0.0
        accuracy = raw_accuracy if raw_accuracy is not None else 0.0
        overkill = raw_overkill if raw_overkill is not None else 1.0
        if raw_miss is not None:
            miss = raw_miss
        else:
            miss = max(0.0, 1.0 - ng_recall)

        view = {
            "miss_rate": miss,
            "overkill_rate": overkill,
            "ng_recall": ng_recall,
            "accuracy": accuracy,
        }

        # Priority ladder against the RELAXED tier. Order per spec:
        # P0 miss -> P2 overkill -> P1 ng_recall -> P4 accuracy. miss (P0) is
        # checked first so it always outranks overkill (P2).
        if miss > config.MISS_RATE_RELAXED_MAX:
            return {
                "label": "OPTIMIZING_P0_MISS_RATE",
                "detail": f"miss_rate={miss:.3f} > {config.MISS_RATE_RELAXED_MAX:.3f} (P0)",
            }
        if overkill > config.OVERKILL_RELAXED_MAX:
            return {
                "label": "MITIGATING_P2_OVERKILL",
                "detail": f"overkill_rate={overkill:.3f} > {config.OVERKILL_RELAXED_MAX:.3f} (P2)",
            }
        if ng_recall < config.NG_RECALL_RELAXED_MIN:
            return {
                "label": "LIFTING_NG_RECALL",
                "detail": f"ng_recall={ng_recall:.3f} < {config.NG_RECALL_RELAXED_MIN:.3f} (P1)",
            }
        if accuracy < config.ACCURACY_RELAXED_MIN:
            return {
                "label": "RAISING_ACCURACY",
                "detail": f"accuracy={accuracy:.3f} < {config.ACCURACY_RELAXED_MIN:.3f} (P4)",
            }

        # Relaxed tier fully met. Distinguish final-tier status.
        if passes_relaxed_acceptance(view) and not passes_final_acceptance(view):
            return {
                "label": "PUSHING_TO_FINAL_TARGET",
                "detail": _final_gap_detail(miss, overkill, ng_recall, accuracy),
            }

        return {
            "label": "HOLDING_ALL_TARGETS_MET",
            "detail": "all relaxed and final targets met — guard against regression",
        }
    except Exception:
        # This helper feeds a never-raise hook and two prompt loaders. On any
        # unexpected failure, fall back to the conservative P0 default.
        return dict(_DEFAULT)


def _final_gap_detail(miss: float, overkill: float, ng_recall: float, accuracy: float) -> str:
    """Dominant open gap to the FINAL tier, in P0 -> P2 -> P1 -> P4 ladder order."""
    if miss > config.MISS_RATE_FINAL_MAX:
        return f"miss_rate={miss:.3f} > {config.MISS_RATE_FINAL_MAX:.3f} (final P0)"
    if overkill > config.OVERKILL_FINAL_MAX:
        return f"overkill_rate={overkill:.3f} > {config.OVERKILL_FINAL_MAX:.3f} (final P2)"
    if ng_recall < config.NG_RECALL_FINAL_MIN:
        return f"ng_recall={ng_recall:.3f} < {config.NG_RECALL_FINAL_MIN:.3f} (final P1)"
    if accuracy < config.ACCURACY_FINAL_MIN:
        return f"accuracy={accuracy:.3f} < {config.ACCURACY_FINAL_MIN:.3f} (final P4)"
    return "relaxed met; closing remaining final-tier gap"


def analytical_state_line(state: Any) -> str:
    """Render the first-line prompt banner for the planner/coder context loaders.

    Uses a precomputed state["analytical_state"] when present (set at the end of
    the previous inner loop); otherwise computes it on the fly from current best
    metrics so the banner is never omitted on the first iteration.
    """
    snapshot = None
    if isinstance(state, Mapping):
        cached = state.get("analytical_state")
        if isinstance(cached, Mapping) and cached.get("label"):
            snapshot = cached
    if snapshot is None:
        snapshot = compute_analytical_state(state)
    return (
        f"ANALYTICAL_STATE: {snapshot.get('label')} — {snapshot.get('detail')}. "
        "Keep this as the active objective."
    )
