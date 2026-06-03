import hashlib
import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.guards.code_validator_agent import (
    check_validation_cache_tool,
    code_validator_tool,
    store_validation_cache_tool,
)
from mle_star_agent.shared.acceptance_scoring import (
    is_acceptance_improvement,
    metrics_view,
    passes_relaxed_acceptance,
)
from mle_star_agent.shared import code_runner, loop_guard, metric_guard
from mle_star_agent.shared.callbacks import (
    count_tokens_callback,
    log_context_size_callback,
    rate_limit_retry_callback,
)
from mle_star_agent.shared.checkpoint_io import save_checkpoint
from mle_star_agent.shared.metrics_parser import (
    AOIMetrics,
    metrics_to_dict,
    parse_error_analysis,
    parse_metrics,
    parse_probe_metrics,
)
from mle_star_agent.shared.diagnosis_scorer import (
    parse_calibration_stats,
    parse_epoch_logs,
    parse_threshold_curve,
    detect_early_collapse,
)
from mle_star_agent.phases.phase2_refinement.ideator_agent import trigger_ideation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_best_pipeline(state: dict) -> None:
    """Persist the authoritative resume snapshot to best_pipeline.json."""
    current_best_score = float(state.get("current_best_score", 0.0) or 0.0)
    best_miss_rate = state.get("best_miss_rate")
    if best_miss_rate is None:
        best_miss_rate = max(0.0, 1.0 - current_best_score)

    save_checkpoint(config.CKPT_BEST_PIPELINE, {
        "outer_iteration":    state.get("outer_iteration", 0),
        "inner_iteration":    state.get("inner_iteration", 0),
        "no_improve_count":   state.get("no_improve_count", 0),
        "current_best_score": current_best_score,
        "best_overkill_rate": state.get("best_overkill_rate", 1.0),
        "best_miss_rate":     float(best_miss_rate),
        "best_accuracy":      state.get("best_accuracy", 0.0),
        "best_f1":            state.get("best_f1", 0.0),
        "best_roc_auc":       state.get("best_roc_auc", 0.0),
        "best_prob_gap":      state.get("best_prob_gap", 0.0),
        "best_pipeline_script": state.get("best_pipeline_script", ""),
        # Preserve Phase 3 progress if it already ran; fall back to Phase 2 best.
        "ensemble_best_score":   state.get("ensemble_best_score", current_best_score),
        "ensemble_best_overkill": state.get("ensemble_best_overkill", state.get("best_overkill_rate", 1.0)),
        "ensemble_best_accuracy": state.get("ensemble_best_accuracy", state.get("best_accuracy", 0.0)),
        "ensemble_best_f1":      state.get("ensemble_best_f1", state.get("best_f1", 0.0)),
        "ensemble_iteration":    state.get("ensemble_iteration", 0),
        "refinement_population": state.get("refinement_population", []),
        "token_count":        state.get("token_count", 0),
        "stop_outer_loop":    state.get("stop_outer_loop", False),
    })


def _clear_ablation_state(state: dict) -> None:
    for key in list(state.keys()):
        if key == "ablation_results" or key.startswith("ablation_result_") or key.startswith("ablation_script_"):
            state.pop(key, None)


def _is_improvement(
    new_ng_recall: float,
    new_overkill: float,
    current_ng_recall: float,
    current_overkill: float,
    new_metrics=None,
    current_metrics=None,
) -> bool:
    """
    Constrained improvement check per spec §8 priority order:
      P0/P1: maximise NG Recall (miss_rate = 1 - ng_recall, same axis)
      P2:    overkill_rate <= OVERKILL_RELAXED_MAX (0.08) is a hard gate

    Rules:
      1. Constrained (overkill <= 0.08) always beats unconstrained.
      2. Both constrained: higher ng_recall wins; tiebreak on lower overkill.
      3. Neither constrained: fall back to ng_recall only (prevents deadlock).
      4. New unconstrained, current constrained: never an improvement.
    """
    if new_metrics is not None and current_metrics is not None:
        return is_acceptance_improvement(new_metrics, current_metrics)

    return is_acceptance_improvement(
        {
            "accuracy": 0.0,
            "ng_recall": new_ng_recall,
            "miss_rate": max(0.0, 1.0 - new_ng_recall),
            "overkill_rate": new_overkill,
            "f1": 0.0,
        },
        {
            "accuracy": 0.0,
            "ng_recall": current_ng_recall,
            "miss_rate": max(0.0, 1.0 - current_ng_recall),
            "overkill_rate": current_overkill,
            "f1": 0.0,
        },
    )


def _normalise_metrics_dict(metrics) -> dict:
    """Return a plain metrics dict from AOIMetrics or an existing mapping."""
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return dict(metrics)
    return metrics_to_dict(metrics)


def _requires_multiseed_confirmation(new_metrics, improved: bool) -> bool:
    """Only spend 3x compute when a single run looks close enough to be decision-worthy."""
    if not improved:
        return False
    metrics = _normalise_metrics_dict(new_metrics)
    if not metrics:
        return False
    return (
        float(metrics.get("overkill_rate", 1.0) or 1.0)
        <= config.MULTISEED_CONFIRMATION_OVERKILL_MAX
    )


def _probe_rejection_reason(probe_metrics: dict | None) -> str | None:
    if not isinstance(probe_metrics, dict):
        return None

    if probe_metrics.get("should_continue") is False:
        return probe_metrics.get("reason") or "probe marked should_continue=false"
    if probe_metrics.get("passed") is False:
        return probe_metrics.get("reason") or "probe marked passed=false"
    if probe_metrics.get("abort") is True:
        return probe_metrics.get("reason") or "probe marked abort=true"

    overkill = probe_metrics.get("overkill_rate")
    if overkill is not None and float(overkill) > config.PROBE_OVERKILL_REJECT_MAX:
        return (
            f"probe overkill_rate={float(overkill):.3f} exceeds "
            f"{config.PROBE_OVERKILL_REJECT_MAX:.2f}"
        )

    recall = probe_metrics.get("ng_recall")
    if recall is not None and float(recall) < config.PROBE_NG_RECALL_REJECT_MIN:
        return (
            f"probe ng_recall={float(recall):.3f} is below "
            f"{config.PROBE_NG_RECALL_REJECT_MIN:.2f}"
        )

    probability_gap = probe_metrics.get("probability_gap")
    if (
        probability_gap is not None
        and float(probability_gap) < config.PROBE_PROBABILITY_GAP_MIN
    ):
        return (
            f"probe probability_gap={float(probability_gap):.3f} is below "
            f"{config.PROBE_PROBABILITY_GAP_MIN:.2f}"
        )

    return None


def _average_metrics_dicts(metric_dicts: list[dict]) -> dict:
    keys = (
        "accuracy", "ng_recall", "miss_rate", "overkill_rate", "f1",
        "avg_latency_ms", "threshold", "ng_count", "g_count", "tp", "tn", "fp", "fn",
        "roc_auc", "prob_gap",
    )
    averaged = {}
    for key in keys:
        values = [float(m.get(key, 0.0) or 0.0) for m in metric_dicts]
        averaged[key] = sum(values) / len(values) if values else 0.0
    for key in ("ng_count", "g_count", "tp", "tn", "fp", "fn"):
        averaged[key] = int(round(averaged[key]))
    return averaged


def _metrics_from_dict(metrics: dict) -> AOIMetrics:
    def value(key: str, default):
        raw = metrics.get(key, default)
        return default if raw is None else raw

    return AOIMetrics(
        accuracy=float(value("accuracy", 0.0)),
        ng_recall=float(value("ng_recall", 0.0)),
        miss_rate=float(value("miss_rate", 1.0)),
        overkill_rate=float(value("overkill_rate", 1.0)),
        f1=float(value("f1", 0.0)),
        avg_latency_ms=float(value("avg_latency_ms", 0.0)),
        threshold=float(value("threshold", 0.5)),
        ng_count=int(value("ng_count", 0)),
        g_count=int(value("g_count", 0)),
        tp=int(value("tp", 0)),
        tn=int(value("tn", 0)),
        fp=int(value("fp", 0)),
        fn=int(value("fn", 0)),
        roc_auc=float(value("roc_auc", 0.0)),
        prob_gap=float(value("prob_gap", 0.0)),
    )


def _prediction_value(prediction: dict, key: str) -> float | None:
    value = prediction.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _verify_prediction_contract(prediction: dict | None, metrics) -> dict:
    """Compare actual metrics to the diagnosis agent's falsifiable prediction."""
    if not isinstance(prediction, dict) or not prediction:
        return {"status": "missing", "failed_constraints": []}
    if metrics is None:
        return {
            "status": "failed",
            "failed_constraints": ["metrics"],
            "prediction": prediction,
            "actual": None,
        }

    actual = metrics_view(metrics)
    failed = []

    overkill_max = _prediction_value(prediction, "expected_overkill_rate_max")
    if overkill_max is not None and actual["overkill_rate"] > overkill_max:
        failed.append("overkill_rate")

    overkill_min = _prediction_value(prediction, "expected_overkill_rate_min")
    if overkill_min is not None and actual["overkill_rate"] < overkill_min:
        failed.append("overkill_rate")

    recall_min = _prediction_value(prediction, "expected_ng_recall_min")
    if recall_min is not None and actual["ng_recall"] < recall_min:
        failed.append("ng_recall")

    miss_max = _prediction_value(prediction, "expected_miss_rate_max")
    if miss_max is not None and actual["miss_rate"] > miss_max:
        failed.append("miss_rate")

    accuracy_min = _prediction_value(prediction, "expected_accuracy_min")
    if accuracy_min is not None and actual["accuracy"] < accuracy_min:
        failed.append("accuracy")

    status = "failed" if failed else "satisfied"
    return {
        "status": status,
        "failed_constraints": failed,
        "prediction": prediction,
        "actual": actual,
    }


def _run_multiseed_confirmation(script: str) -> tuple[AOIMetrics | None, list[dict]]:
    """Run a promising candidate across configured seeds and return averaged metrics."""
    seed_results = []
    for seed in config.MULTISEED_CONFIRMATION_SEEDS:
        result = code_runner.run_script(
            script,
            timeout=config.TIMEOUT_SECONDS,
            env={
                "AOI_RANDOM_SEED": str(seed),
                "PYTHONHASHSEED": str(seed),
                "SEED": str(seed),
            },
        )
        metrics = parse_metrics(result.stdout)
        metrics = metric_guard.guard_metrics(
            metrics, result.duration_ms, context=f"phase2 multiseed seed={seed}"
        )
        seed_results.append({
            "seed": seed,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration_ms": round(result.duration_ms, 1),
            "metrics": metrics_to_dict(metrics) if metrics else None,
            "stderr_tail": result.stderr[-1000:],
        })
    successful = [r["metrics"] for r in seed_results if r.get("metrics") is not None]
    if len(successful) != len(config.MULTISEED_CONFIRMATION_SEEDS):
        return None, seed_results
    return _metrics_from_dict(_average_metrics_dicts(successful)), seed_results


def _update_refinement_population(
    state: dict,
    script: str,
    metrics,
    current_metrics: dict,
    improved: bool,
    outer_iteration: int,
    inner_iteration: int,
) -> None:
    """Keep a small archive of useful candidates so refinement is not a single chain."""
    metrics_dict = _normalise_metrics_dict(metrics)
    if not metrics_dict:
        return
    new_overkill = float(metrics_dict.get("overkill_rate", 1.0) or 1.0)
    current_overkill = float(current_metrics.get("overkill_rate", 1.0) or 1.0)
    if improved:
        archive_reason = "accepted_improvement"
    elif new_overkill < current_overkill:
        archive_reason = "lower_overkill_candidate"
    else:
        return

    script_hash = hashlib.sha256(script.encode("utf-8", errors="replace")).hexdigest()
    entry = {
        "outer": outer_iteration,
        "inner": inner_iteration,
        "script_sha256": script_hash,
        "script": script,
        "metrics": metrics_dict,
        "archive_reason": archive_reason,
    }
    population = [
        p for p in list(state.get("refinement_population", []) or [])
        if p.get("script_sha256") != script_hash
    ]
    population.append(entry)
    population.sort(key=lambda p: (
        float(p.get("metrics", {}).get("overkill_rate", 1.0) or 1.0),
        float(p.get("metrics", {}).get("miss_rate", 1.0) or 1.0),
        -float(p.get("metrics", {}).get("accuracy", 0.0) or 0.0),
        -float(p.get("metrics", {}).get("ng_recall", 0.0) or 0.0),
    ))
    state["refinement_population"] = population[:config.REFINEMENT_POPULATION_MAX]


# ---------------------------------------------------------------------------
# Main evaluation FunctionTool
# ---------------------------------------------------------------------------

def evaluate_and_update_fn(tool_context) -> str:
    """
    Run `script`, parse METRICS, update loop state, manage loop exits.

    Exit logic (all via tool_context.actions.escalate inside this FunctionTool):

    Priority 1 — mid-inner early-stop (no_improve or token budget exceeded):
        Set stop_outer_loop = True, escalate.  Outer loop caught by ablation_flag_checker.

    Priority 2 — inner loop cap reached:
        Increment outer_iteration, reset inner_iteration.
        If outer exit condition now met: set stop_outer_loop = True.
        Escalate to exit inner_loop_agent.  Outer loop continues unless flag is set.

    Priority 3 — normal inner continuation:
        Save checkpoint, return metrics summary.  Inner loop continues.
    """
    # ---- read script from state (written by save_validated_script_fn) ----
    script = tool_context.state.get("current_script", "")

    # ---- read current counters ----
    n = int(tool_context.state.get("outer_iteration", 0))
    m = int(tool_context.state.get("inner_iteration", 0))
    current_best         = float(tool_context.state.get("current_best_score", 0.0))
    current_best_overkill = float(tool_context.state.get("best_overkill_rate", 1.0))
    current_best_miss = tool_context.state.get("best_miss_rate")
    if current_best_miss is None:
        current_best_miss = max(0.0, 1.0 - current_best)
    current_best_miss = float(current_best_miss)
    current_best_accuracy = float(tool_context.state.get("best_accuracy", 0.0))
    current_best_f1 = float(tool_context.state.get("best_f1", 0.0))
    no_improve           = int(tool_context.state.get("no_improve_count", 0))

    # ---- debug pre-check: fast smoke run before paying for the full run ----
    # Patches the script to max_epochs=1 + 5% data and caps the timeout at
    # DEBUG_CHECK_TIMEOUT_SECONDS. If it fails, the script is broken and there is
    # no point spending hours on the full run — record the failure and bail.
    logger.info("Evaluator debug pre-check (outer=%d, inner=%d)", n, m)
    debug_result = code_runner.run_script(
        script,
        timeout=config.TIMEOUT_SECONDS,
        env={"AOI_RANDOM_SEED": "42", "PYTHONHASHSEED": "42", "SEED": "42"},
        debug_mode=True,
    )
    if debug_result.returncode != 0:
        logger.warning(
            "Debug pre-check failed (outer=%d, inner=%d, rc=%d) — skipping full run",
            n, m, debug_result.returncode,
        )
        m_next = m + 1
        tool_context.state["inner_iteration"] = m_next
        no_improve += 1
        tool_context.state["no_improve_count"] = no_improve

        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        attempt_data = {
            "outer_iteration": n,
            "inner_iteration": m_next,
            "returncode":      debug_result.returncode,
            "timed_out":       debug_result.timed_out,
            "duration_ms":     round(debug_result.duration_ms, 1),
            "improved":        False,
            "failure_reason":  "debug_check_failed",
            "new_score":       0.0,
            "new_overkill":    1.0,
            "current_best_score":   float(tool_context.state.get("current_best_score", 0.0)),
            "current_best_overkill": float(tool_context.state.get("best_overkill_rate", 1.0)),
            "no_improve_count":   no_improve,
            "metrics":         None,
            "stdout_tail":     debug_result.stdout[-3000:],
            "stderr_tail":     debug_result.stderr[-1000:],
        }
        save_checkpoint(config.ckpt_refinement(n, m), attempt_data)
        return (
            f"DEBUG CHECK FAILED (outer={n}, inner={m}): script exited with "
            f"returncode={debug_result.returncode} in the accelerated smoke run; "
            f"full run skipped. inner_iteration now {m_next}, "
            f"no_improve_count now {no_improve}."
        )

    # ---- execute the script ----
    logger.info("Evaluator running script (outer=%d, inner=%d)", n, m)
    result = code_runner.run_script(
        script,
        timeout=config.TIMEOUT_SECONDS,
        env={"AOI_RANDOM_SEED": "42", "PYTHONHASHSEED": "42", "SEED": "42"},
    )
    parsed_metrics = parse_metrics(result.stdout)
    # Persistence-boundary guard: a degenerate refinement run must not be written
    # as a valid candidate metric (it would poison acceptance + diagnosis).
    metrics = metric_guard.guard_metrics(
        parsed_metrics, result.duration_ms, context=f"phase2 refinement outer={n} inner={m}"
    )
    probe_metrics = parse_probe_metrics(result.stdout)
    probe_rejection_reason = _probe_rejection_reason(probe_metrics)
    if probe_metrics:
        tool_context.state["latest_probe_metrics"] = probe_metrics
    else:
        tool_context.state.pop("latest_probe_metrics", None)
    error_analysis = parse_error_analysis(result.stdout, metrics=metrics)

    # Parse training diagnostic signals (P1)
    calibration_stats = parse_calibration_stats(result.stdout)
    epoch_logs = parse_epoch_logs(result.stdout)
    threshold_curve = parse_threshold_curve(result.stdout)
    early_collapse = detect_early_collapse(epoch_logs) if epoch_logs else None
    if calibration_stats:
        tool_context.state["latest_calibration_stats"] = calibration_stats
    if epoch_logs:
        tool_context.state["latest_epoch_logs"] = epoch_logs
    if threshold_curve:
        tool_context.state["latest_threshold_curve"] = threshold_curve
    if early_collapse and early_collapse.get("detected"):
        logger.warning("Early collapse detected: %s", early_collapse["pattern"])

    probe_rejected = result.returncode == 0 and probe_rejection_reason is not None
    run_ok      = result.returncode == 0 and metrics is not None and not probe_rejected
    new_score   = float(metrics.ng_recall)    if metrics else 0.0
    new_overkill = float(metrics.overkill_rate) if metrics else 1.0

    # Clear instrumentation flag once the script successfully emits per-sample evidence
    if error_analysis.get("available"):
        tool_context.state.pop("error_analysis_instrumentation_required", None)
        tool_context.state.pop("error_analysis_repair_attempted", None)
        tool_context.state.pop("error_analysis_blocked", None)

    # ---- decide improvement (spec §8: P0/P1=ng_recall, P2=overkill constraint) ----
    current_metrics = {
        "accuracy": current_best_accuracy,
        "ng_recall": current_best,
        "miss_rate": current_best_miss,
        "overkill_rate": current_best_overkill,
        "f1": current_best_f1,
    }
    improved = run_ok and _is_improvement(
        new_score,
        new_overkill,
        current_best,
        current_best_overkill,
        new_metrics=metrics,
        current_metrics=current_metrics,
    )
    multiseed_results = []
    if run_ok and _requires_multiseed_confirmation(metrics, improved):
        averaged_metrics, multiseed_results = _run_multiseed_confirmation(script)
        tool_context.state["latest_multiseed_confirmation"] = multiseed_results
        if averaged_metrics is None:
            improved = False
        else:
            metrics = averaged_metrics
            new_score = float(metrics.ng_recall)
            new_overkill = float(metrics.overkill_rate)
            improved = _is_improvement(
                new_score,
                new_overkill,
                current_best,
                current_best_overkill,
                new_metrics=metrics,
                current_metrics=current_metrics,
            )
    else:
        tool_context.state.pop("latest_multiseed_confirmation", None)

    if run_ok and metrics is not None:
        _update_refinement_population(
            tool_context.state,
            script=script,
            metrics=metrics,
            current_metrics=current_metrics,
            improved=improved,
            outer_iteration=n,
            inner_iteration=m,
        )

    diagnosis_report = tool_context.state.get("diagnosis_report") or {}
    if not isinstance(diagnosis_report, dict):
        diagnosis_report = {}
    prediction_verification = _verify_prediction_contract(
        diagnosis_report.get("prediction"),
        metrics if run_ok else parsed_metrics,
    )
    tool_context.state["latest_prediction_verification"] = prediction_verification
    prediction_failed = prediction_verification.get("status") == "failed"
    if prediction_failed and metrics is not None and not passes_relaxed_acceptance(metrics):
        improved = False

    if improved:
        tool_context.state["current_best_score"]   = new_score
        tool_context.state["best_overkill_rate"]   = new_overkill
        tool_context.state["best_accuracy"]         = float(metrics.accuracy)
        tool_context.state["best_f1"]               = float(metrics.f1)
        tool_context.state["best_miss_rate"]        = float(metrics.miss_rate)
        tool_context.state["best_roc_auc"]          = float(metrics.roc_auc)
        tool_context.state["best_prob_gap"]         = float(metrics.prob_gap)
        tool_context.state["best_pipeline_script"] = script
        tool_context.state["no_improve_count"]     = 0
        no_improve = 0
        logger.info(
            "New best: ng_recall=%.4f overkill=%.4f (was ng_recall=%.4f overkill=%.4f)",
            new_score, new_overkill, current_best, current_best_overkill,
        )
    else:
        no_improve += 1
        tool_context.state["no_improve_count"] = no_improve

    failure_reason = (
        "accepted" if improved else
        "diagnosis_prediction_failed" if prediction_failed else
        "probe_rejected" if probe_rejected else
        "execution_failed" if not run_ok else
        "overkill_regression" if (new_overkill > current_best_overkill) else
        "recall_regression" if (new_score < current_best) else
        "no_improvement"
    )

    # ---- increment inner counter ----
    m_next = m + 1
    tool_context.state["inner_iteration"] = m_next

    # ---- persist per-attempt checkpoint ----
    # Record m_next (post-increment) so resume reads the same value as state["inner_iteration"]
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    error_analysis_path = config.ckpt_error_analysis(n, m)
    error_analysis_checkpoint = {
        "outer_iteration": n,
        "inner_iteration": m,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "metrics": metrics_to_dict(metrics) if metrics else None,
        "threshold_curve": threshold_curve,
        "probe_metrics": probe_metrics,
        "probe_rejection_reason": probe_rejection_reason,
        **error_analysis,
    }
    save_checkpoint(error_analysis_path, error_analysis_checkpoint)
    tool_context.state["latest_error_analysis"] = error_analysis_checkpoint
    tool_context.state["latest_error_analysis_path"] = str(error_analysis_path)

    attempt_data = {
        "outer_iteration": n,
        "inner_iteration": m_next,
        "returncode":      result.returncode,
        "timed_out":       result.timed_out,
        "duration_ms":     round(result.duration_ms, 1),
        "improved":        improved,
        "failure_reason":  failure_reason,
        "new_score":       new_score,
        "new_overkill":    new_overkill,
        "current_best_score":   tool_context.state["current_best_score"],
        "current_best_overkill": tool_context.state.get("best_overkill_rate", 1.0),
        "no_improve_count":   no_improve,
        "metrics":         metrics_to_dict(metrics) if metrics else None,
        "error_analysis_checkpoint": str(error_analysis_path),
        "error_analysis_available": error_analysis.get("available", False),
        "threshold_curve": threshold_curve,
        "probe_metrics": probe_metrics,
        "probe_rejection_reason": probe_rejection_reason,
        "prediction_verification": prediction_verification,
        "multiseed_confirmation": multiseed_results,
        "refinement_population_count": len(tool_context.state.get("refinement_population", []) or []),
        "stdout_tail":     result.stdout[-3000:],
        "stderr_tail":     result.stderr[-1000:],
    }
    save_checkpoint(config.ckpt_refinement(n, m), attempt_data)

    # Append to tried_approaches memory (P4)
    tried = list(tool_context.state.get("tried_approaches", []) or [])
    plan = tool_context.state.get("refinement_plan") or {}
    selected_strategy = tool_context.state.get("selected_refinement_strategy", "")
    tried.append({
        "outer": n, "inner": m,
        "target_component": plan.get("target_component", "unknown"),
        "changes_summary": plan.get("changes_summary", ""),
        "selected_strategy": selected_strategy,
        "strategy_fingerprint": tool_context.state.get("selected_strategy_fingerprint"),
        "result": {
            "ng_recall": round(new_score, 4),
            "miss_rate": round(float(metrics.miss_rate), 4) if metrics else 1.0,
            "overkill": round(new_overkill, 4),
            "accuracy": round(float(metrics.accuracy), 4) if metrics else 0.0,
            "improved": improved,
        },
        "prediction_verification": prediction_verification,
        "failure_reason": failure_reason,
    })
    tool_context.state["tried_approaches"] = tried
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(config.CKPT_TRIED_APPROACHES, {"tried_approaches": tried})

    # Build a human-readable metrics line for the return value
    if probe_rejected:
        metrics_line = f"PROBE_REJECTED ({probe_rejection_reason})"
    elif metrics:
        metrics_line = (
            f"accuracy={metrics.accuracy:.3f}  ng_recall={metrics.ng_recall:.3f}  "
            f"miss_rate={metrics.miss_rate:.3f}  overkill={metrics.overkill_rate:.3f}  "
            f"f1={metrics.f1:.3f}  roc_auc={metrics.roc_auc:.3f}  prob_gap={metrics.prob_gap:.3f}  "
            f"threshold={metrics.threshold}  "
            f"({'NEW BEST' if improved else 'no improvement'})"
        )
    else:
        metrics_line = (
            f"FAILED (rc={result.returncode}, timed_out={result.timed_out})  "
            f"stderr: {result.stderr[-200:]}"
        )

    summary_prefix = (
        f"[outer={n}, inner={m}]  {metrics_line}  |  "
        f"best={tool_context.state['current_best_score']:.4f}  "
        f"no_improve={no_improve}"
    )

    # ---- Priority 1a: high-overkill bounded restart ----
    # Fires when overkill >= 0.50 after INNER_STAGNATION_MAX_HIGH_OVERKILL attempts,
    # regardless of whether those attempts counted as "improvements" on other metrics.
    if loop_guard.should_restart_for_high_overkill(tool_context.state):
        n_next = n + 1
        tool_context.state["outer_iteration"] = n_next
        tool_context.state["inner_iteration"] = 0
        tool_context.state["no_improve_count"] = 0
        tool_context.state["force_fresh_ablation"] = True
        _clear_ablation_state(tool_context.state)
        _save_best_pipeline(tool_context.state)
        tool_context.actions.escalate = True
        current_overkill = tool_context.state.get("best_overkill_rate", 1.0)
        logger.info(
            "High-overkill bounded restart: overkill=%.3f >= %.2f after %d inner attempts — "
            "restarting at outer=%d.",
            current_overkill, config.HIGH_OVERKILL_STAGNATION_THRESHOLD,
            config.INNER_STAGNATION_MAX_HIGH_OVERKILL, n_next,
        )
        return (
            f"{summary_prefix}\n"
            f"HIGH_OVERKILL_RESTART: overkill={current_overkill:.3f} has not dropped below "
            f"{config.HIGH_OVERKILL_STAGNATION_THRESHOLD} after "
            f"{config.INNER_STAGNATION_MAX_HIGH_OVERKILL} inner attempts. "
            "Further refinement on this component is not useful at this overkill level — "
            "fresh ablation needed to identify a different high-impact component. "
            f"outer_iteration advanced to {n_next}, inner_iteration reset to 0, "
            "ablation state cleared. Escalating inner loop; outer loop continues."
        )

    # ---- Priority 1b: below-relaxed inner stagnation restart ----
    if loop_guard.should_restart_inner_for_stagnation(tool_context.state):
        n_next = n + 1
        tool_context.state["outer_iteration"] = n_next
        tool_context.state["inner_iteration"] = 0
        tool_context.state["no_improve_count"] = 0
        tool_context.state["force_fresh_ablation"] = True
        _clear_ablation_state(tool_context.state)
        # Dynamic ideation injection (Fix #1): the idea pool has gone stale — the
        # inner loop stalled below relaxed acceptance. Pull fresh, failure-mode-keyed
        # technique hints from arXiv into state["retrieved_technique_hints"] BEFORE
        # escalating, so the next diagnosis/planning cycle has new levers to try.
        ideation_status = trigger_ideation(tool_context)
        _save_best_pipeline(tool_context.state)
        tool_context.actions.escalate = True
        logger.info(
            "Inner stagnation hit below relaxed acceptance — restarting at outer=%d.",
            n_next,
        )
        return (
            f"{summary_prefix}\n"
            f"INNER_STAGNATION: no improvement for "
            f"{config.INNER_STAGNATION_MAX_UNCONSTRAINED} below-relaxed attempts. "
            f"outer_iteration advanced to {n_next}, inner_iteration reset to 0, "
            "and ablation state cleared. Escalating inner loop; outer loop continues.\n"
            f"{ideation_status}"
        )

    # ---- Priority 2: mid-inner early-stop ----
    # Check acceptance-aware no-improvement and token budget against the UPDATED state.
    # ADK State supports .get() directly — avoid accessing private ._value.
    if (
        loop_guard.should_stop_for_no_improvement(tool_context.state)
        or tool_context.state.get("token_count", 0) >= config.TOKEN_BUDGET
    ):
        tool_context.state["stop_outer_loop"] = True
        _save_best_pipeline(tool_context.state)
        reason = (
            "accepted pipeline reached no-improvement patience"
            if loop_guard.should_stop_for_no_improvement(tool_context.state)
            else "token budget exhausted"
        )
        tool_context.actions.escalate = True
        logger.info("Early-stop triggered (%s) — outer loop will exit via flag.", reason)
        return (
            f"{summary_prefix}\n"
            f"EARLY_STOP ({reason}): stop_outer_loop=True set. "
            "Escalating inner loop; outer loop exits on next ablation_flag_checker call."
        )

    # ---- Priority 3: inner loop cap ----
    if loop_guard.should_exit_inner(tool_context.state):
        # Advance to next outer iteration
        n_next = n + 1
        tool_context.state["outer_iteration"] = n_next
        tool_context.state["inner_iteration"] = 0

        # Check if outer exit condition is now triggered
        if loop_guard.should_exit_outer(tool_context.state):
            tool_context.state["stop_outer_loop"] = True
            _save_best_pipeline(tool_context.state)
            tool_context.actions.escalate = True
            logger.info(
                "Inner cap hit; outer exit condition met after advancing to outer=%d.", n_next
            )
            return (
                f"{summary_prefix}\n"
                f"INNER_CAP: inner loop complete. outer_iteration advanced to {n_next}. "
                "Outer exit condition met — stop_outer_loop=True set. Escalating."
            )

        _save_best_pipeline(tool_context.state)
        tool_context.actions.escalate = True
        logger.info("Inner cap hit — escalating; outer=%d continues.", n_next)
        return (
            f"{summary_prefix}\n"
            f"INNER_CAP: inner loop complete. outer_iteration advanced to {n_next}. "
            "Escalating inner loop; outer loop continues."
        )

    # ---- Priority 4: normal continuation ----
    _save_best_pipeline(tool_context.state)
    return f"{summary_prefix}\nCONTINUE: inner loop proceeding (inner_iteration now {m_next})."


_evaluate_tool = FunctionTool(func=evaluate_and_update_fn)

# ---------------------------------------------------------------------------
# FunctionTool: commit the validated script to state before evaluation
# Decouples the LLM's text-extraction step from the execution step so that
# evaluate_and_update_fn always reads from state, not from the LLM argument.
# ---------------------------------------------------------------------------

def save_validated_script_fn(tool_context, script: str) -> str:
    """
    Write the (possibly validator-corrected) script to state["current_script"]
    so that evaluate_and_update_fn reads an authoritative value from state.
    Call this after code_validator_agent returns and before evaluate_and_update_fn.
    """
    tool_context.state["current_script"] = script
    return "Validated script committed to state['current_script']."


_save_validated_script_tool = FunctionTool(func=save_validated_script_fn)


def load_current_script_fn(tool_context) -> str:
    """
    Return state["current_script"] (the script written by refinement_coder_agent)
    into THIS turn. Call this FIRST so the cache-check and validation steps have the
    script text without relying on conversation history (include_contents="none").
    """
    script = tool_context.state.get("current_script", "")
    if not script:
        return "ERROR: current_script not found in state."
    return script


_load_current_script_tool = FunctionTool(func=load_current_script_fn)

# ---------------------------------------------------------------------------
# Agent instruction
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are the Evaluator Agent for Phase 2 Refinement.

Your role is to validate the refined script, execute it, and manage both
the inner and outer loop termination conditions.

---
## STEP 1 — Load the current script

Call `load_current_script_fn` FIRST. It returns the script written by
`refinement_coder_agent` in the previous step. Do NOT rely on conversation history
for the script text — use the value this tool returns for all subsequent steps.

---
## STEP 2 — Validate

First call `check_validation_cache_fn` with the script text returned by `load_current_script_fn`.
- If it returns "CACHE_HIT: VALIDATED":
  The script was already validated by the coder agent — skip `code_validator_agent`.
  Call `save_validated_script_fn` with the original script and proceed to STEP 3.
- If it returns "CACHE_HIT: VALIDATION_FAILED":
  The script is known-broken from the coder's validation pass — skip `code_validator_agent`.
  Call `save_validated_script_fn` with the original script, then proceed to STEP 3.
  `evaluate_and_update_fn` will run the script, detect returncode != 0, record no improvement,
  and advance the loop counters — this is the correct outcome for a broken script.
- If it returns "CACHE_MISS":
  Call `code_validator_agent` with the script text.
  - If it returns "VALIDATED_SCRIPT:": extract the corrected script that follows.
    Call `store_validation_cache_fn(corrected_script, "VALIDATED")`.
  - If it returns "VALIDATION_FAILED": use the original script unchanged.
    Call `store_validation_cache_fn(original_script, "VALIDATION_FAILED")`.
  Then call `save_validated_script_fn` with the final script (corrected or original).

---
## STEP 3 — Evaluate and manage loops

Call `evaluate_and_update_fn` with no arguments — it reads the script directly
from `state["current_script"]` (committed in the previous step).

This tool:
  - Runs the script and parses METRICS
  - Parses PROBE_METRICS. If the script exits after a failed cheap probe, records
    `failure_reason=probe_rejected` and advances the loop without treating it as
    an execution crash.
  - Parses deterministic per-sample ERROR_ANALYSIS / PREDICTIONS output when present
    and saves checkpoints/error_analysis_[N]_[M].json. If the script omits per-sample
    evidence, it still saves a checkpoint with available=false so the next agent can
    distinguish missing evidence from a clean run.
  - Scores per spec §8 priority: overkill_rate <= 8% is a hard gate (P2), then maximise
    ng_recall (P0/P1). A constrained solution always beats an unconstrained one.
  - Updates state["current_best_score"], state["best_overkill_rate"], and
    state["best_pipeline_script"] if improved
  - Increments inner_iteration and no_improve_count as appropriate
  - Saves checkpoints/refinement_[N]_[M].json (per attempt)
  - Saves checkpoints/best_pipeline.json (resume snapshot)
  - Handles both inner loop exit (escalate) and outer loop flag (stop_outer_loop)

---
## STEP 4 — Report

After `evaluate_and_update_fn` returns, report:
1. The key metrics: accuracy, ng_recall, miss_rate, overkill_rate, f1, threshold
2. Whether the script improved the best score
3. Current loop counters: outer_iteration, inner_iteration, no_improve_count
4. The exit decision (CONTINUE / INNER_CAP / EARLY_STOP) as returned by the tool

Do NOT call any other tools. Do NOT attempt to modify loop counters manually.
All loop state management is handled inside `evaluate_and_update_fn`.
"""

# ---------------------------------------------------------------------------
# Evaluator agent
# ---------------------------------------------------------------------------

evaluator_agent = LlmAgent(
    name="evaluator_agent",
    model=config.MODEL,
    description=(
        "Validates and runs the refined script, updates the best pipeline on improvement, "
        "manages inner/outer loop termination. Saves refinement_{{N}}_{{M}}.json and "
        "best_pipeline.json. Escalates via FunctionTool when loop caps or no-improve "
        "limits are reached."
    ),
    instruction=_INSTRUCTION,
    tools=[_load_current_script_tool, _evaluate_tool, _save_validated_script_tool, code_validator_tool, check_validation_cache_tool, store_validation_cache_tool],
    include_contents="none",
    before_model_callback=log_context_size_callback,
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
