from mle_star_agent import config
from mle_star_agent.shared.acceptance_scoring import passes_relaxed_acceptance, passes_final_acceptance


def should_exit_inner(state: dict) -> bool:
    return state.get("inner_iteration", 0) >= config.INNER_LOOP_MAX


def should_exit_outer(state: dict) -> bool:
    if state.get("outer_iteration", 0) >= config.OUTER_LOOP_MAX:
        return True
    if should_stop_for_no_improvement(state):
        return True
    if state.get("token_count", 0) >= config.TOKEN_BUDGET:
        return True
    return False


def best_metrics_from_state(state: dict) -> dict:
    ng_recall = float(state.get("current_best_score", 0.0) or 0.0)
    saved_miss_rate = state.get("best_miss_rate")
    miss_rate = (
        float(saved_miss_rate)
        if saved_miss_rate is not None
        else max(0.0, 1.0 - ng_recall)
    )
    return {
        "accuracy": float(state.get("best_accuracy", 0.0) or 0.0),
        "ng_recall": ng_recall,
        "miss_rate": miss_rate,
        "overkill_rate": float(state.get("best_overkill_rate", 1.0) or 1.0),
        "f1": float(state.get("best_f1", 0.0) or 0.0),
    }


def should_stop_for_no_improvement(state: dict) -> bool:
    no_improve = state.get("no_improve_count", 0)
    metrics = best_metrics_from_state(state)
    if passes_final_acceptance(metrics):
        return no_improve >= config.NO_IMPROVE_MAX
    if passes_relaxed_acceptance(metrics):
        return no_improve >= config.NO_IMPROVE_MAX_CONSTRAINED
    return False


def should_restart_inner_for_stagnation(state: dict) -> bool:
    """Restart outer diagnosis when below relaxed acceptance and progress is stuck."""
    no_improve = state.get("no_improve_count", 0)
    metrics = best_metrics_from_state(state)
    if passes_relaxed_acceptance(metrics):
        return False
    return no_improve >= config.INNER_STAGNATION_MAX_UNCONSTRAINED


def should_restart_for_high_overkill(state: dict) -> bool:
    """
    Restart outer diagnosis when overkill is catastrophically high (>= 0.50) and
    we have already burned INNER_STAGNATION_MAX_HIGH_OVERKILL inner iterations without
    dropping it below the threshold — regardless of whether those attempts technically
    counted as 'improvements' on other metrics.

    Uses inner_iteration (post-increment) as a wall-clock proxy for attempts elapsed
    in the current outer cycle.  Resets naturally when the outer loop advances.
    """
    metrics = best_metrics_from_state(state)
    if passes_relaxed_acceptance(metrics):
        return False
    if metrics["overkill_rate"] < config.HIGH_OVERKILL_STAGNATION_THRESHOLD:
        return False
    # inner_iteration is already incremented after each completed run
    return state.get("inner_iteration", 0) >= config.INNER_STAGNATION_MAX_HIGH_OVERKILL


def should_exit_ensemble(state: dict) -> bool:
    return state.get("ensemble_iteration", 0) >= config.ENSEMBLE_LOOP_MAX
