"""Plateau-triggered warm-restart refinement operator.

When the inner loop stagnates, this helper arms one bounded optimizer/scheduler
jolt of the current best script before the evaluator abandons the approach and
injects fresh ideas. The actual code generation still flows through the normal
planner -> refinement_coder -> evaluator path, so validation, metrics, checkpoint,
and tried_approaches recording remain unchanged.
"""

import hashlib

from mle_star_agent.shared import loop_guard

WARM_RESTART_DIRECTIVE_MARKER = "plateau_warm_restart"
WARM_RESTART_TARGET_COMPONENT = "optimizer/lr-schedule"


def best_script_sha(state: dict) -> str:
    script = state.get("best_pipeline_script", "") or ""
    return hashlib.sha256(script.encode("utf-8", errors="replace")).hexdigest()[:8]


def warm_restart_fingerprint(best_sha: str) -> dict:
    sha = (best_sha or "unknown").strip()[:8] or "unknown"
    return {
        "target_component": WARM_RESTART_TARGET_COMPONENT,
        "mechanism_class": f"warm_restart_{sha}",
    }


def is_warm_restart_fingerprint(fingerprint: dict | None) -> bool:
    if not isinstance(fingerprint, dict):
        return False
    target = str(fingerprint.get("target_component", "")).strip().lower()
    mechanism = str(fingerprint.get("mechanism_class", "")).strip().lower()
    return target == WARM_RESTART_TARGET_COMPONENT and mechanism.startswith("warm_restart_")


def should_trigger_warm_restart(state: dict) -> bool:
    if not loop_guard.should_restart_inner_for_stagnation(state):
        return False
    sha = best_script_sha(state)
    return not bool(state.get(f"warm_restart_attempted_best_{sha}"))


def arm_warm_restart_if_eligible(state: dict) -> str:
    if not loop_guard.should_restart_inner_for_stagnation(state):
        return "WARM_RESTART_NOT_ARMED: inner loop has not stagnated."

    sha = best_script_sha(state)
    flag = f"warm_restart_attempted_best_{sha}"
    if state.get(flag):
        return (
            "WARM_RESTART_NOT_ARMED: warm restart already attempted for "
            f"current best script {sha}."
        )

    state["pending_warm_restart"] = True
    state["warm_restart_best_sha"] = sha
    state[flag] = True
    return (
        f"WARM_RESTART_ARMED: current best script {sha} will be retried with a "
        "bounded optimizer/lr-schedule warm restart before ideation."
    )


def build_warm_restart_directive(best_sha: str) -> str:
    sha = (best_sha or "unknown").strip()[:8] or "unknown"
    return (
        f"{WARM_RESTART_DIRECTIVE_MARKER}: apply an SGDR-style warm restart to "
        f"the CURRENT best script sha={sha}. Keep the existing model, data split, "
        "loss family, threshold policy, DRY_RUN handling, "
        "`epochs = DRY_RUN_EPOCHS if DRY_RUN else 20`, early stopping, EPOCH_LOG, "
        "PROBE_METRICS, CALIBRATION_STATS, THRESHOLD_CURVE, PREDICTIONS, and "
        "METRICS output intact. Re-raise the optimizer base LR for a new cycle "
        "using `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` "
        "(small-data-safe T_0/T_mult/eta_min), call scheduler.step() at the "
        "correct point, and optionally re-initialize only the final classification "
        "head if the script has a clearly isolated head. Add "
        f"`OPTIMIZER_LR_SCHEDULE_VARIANT = \"warm_restart_{sha}\"` near the "
        "optimizer block so the attempt is visible in logs and diffs."
    )


__all__ = [
    "WARM_RESTART_DIRECTIVE_MARKER",
    "WARM_RESTART_TARGET_COMPONENT",
    "arm_warm_restart_if_eligible",
    "best_script_sha",
    "build_warm_restart_directive",
    "is_warm_restart_fingerprint",
    "should_trigger_warm_restart",
    "warm_restart_fingerprint",
]
