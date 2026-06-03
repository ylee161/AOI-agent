import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.guards.code_validator_agent import (
    check_validation_cache_tool,
    code_validator_tool,
    store_validation_cache_tool,
)
from mle_star_agent.shared.acceptance_scoring import is_acceptance_improvement, metrics_view
from mle_star_agent.shared import code_runner, metric_guard
from mle_star_agent.shared.callbacks import (
    count_tokens_callback,
    log_context_size_callback,
    rate_limit_retry_callback,
)
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint, save_checkpoint
from mle_star_agent.shared.metrics_parser import metrics_to_dict, parse_metrics
from mle_star_agent.shared.diagnosis_scorer import parse_calibration_stats, parse_epoch_logs

logger = logging.getLogger(__name__)


def _record_tried_ensemble_approach(
    state: dict, n: int, metrics, improved: bool, run_ok: bool, failure_reason: Optional[str] = None
) -> None:
    # Use the checkpoint file as the single source of truth to avoid double-counting.
    # State["tried_ensemble_approaches"] is always synced from the checkpoint at the
    # end of this function, so merging both sources would duplicate prior entries.
    tried: list = []
    if checkpoint_exists(config.CKPT_TRIED_ENSEMBLE_APPROACHES):
        checkpoint = load_checkpoint(config.CKPT_TRIED_ENSEMBLE_APPROACHES)
        tried = list(checkpoint.get("tried_ensemble_approaches", []) or [])

    strategy = state.get("ensemble_strategy") or {}
    tried.append({
        "ensemble_iteration": n,
        "strategy_name": strategy.get("strategy_name", "unknown"),
        "combination_method": strategy.get("combination_method", ""),
        "strategy_fingerprint": strategy.get("strategy_fingerprint"),
        "result": {
            "ng_recall": round(float(metrics.ng_recall), 4) if metrics else 0.0,
            "miss_rate": round(float(metrics.miss_rate), 4) if metrics else 1.0,
            "overkill": round(float(metrics.overkill_rate), 4) if metrics else 1.0,
            "accuracy": round(float(metrics.accuracy), 4) if metrics else 0.0,
            "improved": improved,
        },
        "failure_reason": (
            "accepted" if improved else
            "execution_failed" if not run_ok else
            failure_reason or "degenerate_or_no_improvement"
        ),
    })
    state["tried_ensemble_approaches"] = tried
    save_checkpoint(config.CKPT_TRIED_ENSEMBLE_APPROACHES, {
        "tried_ensemble_approaches": tried,
    })

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save_best_ensemble(
    n: int, script: str, score: float, overkill: float, metrics_dict: Optional[dict],
    calibration_stats: Optional[dict] = None,
) -> None:
    """Persist the best ensemble checkpoint (ensemble.json)."""
    save_checkpoint(config.CKPT_ENSEMBLE, {
        "ensemble_iteration":       n,
        "ensemble_best_score":      score,
        "ensemble_best_overkill":   overkill,
        "ensemble_best_accuracy":   (metrics_dict or {}).get("accuracy", 0.0),
        "ensemble_best_f1":         (metrics_dict or {}).get("f1", 0.0),
        "ensemble_script":          script,
        "metrics":                  metrics_dict,
        "calibration_stats":        calibration_stats,
    })


def _is_ensemble_improvement(
    new_ng_recall: float,
    new_overkill: float,
    current_ng_recall: float,
    current_overkill: float,
    new_metrics=None,
    current_metrics=None,
) -> bool:
    """Same constrained scoring as Phase 2 evaluator (spec §8)."""
    candidate = metrics_view(new_metrics) if new_metrics is not None else {
        "accuracy": 0.0,
        "ng_recall": new_ng_recall,
        "miss_rate": max(0.0, 1.0 - new_ng_recall),
        "overkill_rate": new_overkill,
        "f1": 0.0,
    }
    # Reject all-NG / degenerate ensembles (classify everything as NG)
    if candidate["ng_recall"] >= 1.0 and candidate["overkill_rate"] >= 1.0:
        return False
    # Reject genuinely bad recall
    if candidate["ng_recall"] <= 0.50:
        return False

    current = metrics_view(current_metrics) if current_metrics is not None else {
        "accuracy": 0.0,
        "ng_recall": current_ng_recall,
        "miss_rate": max(0.0, 1.0 - current_ng_recall),
        "overkill_rate": current_overkill,
        "f1": 0.0,
    }
    if _has_overkill_regression(candidate, current):
        return False

    if new_metrics is not None and current_metrics is not None:
        return is_acceptance_improvement(new_metrics, current_metrics)

    return is_acceptance_improvement(candidate, current)


def _has_overkill_regression(new_metrics, current_metrics, tolerance: float = 1e-9) -> bool:
    """Return True when an ensemble worsens false rejects on G samples."""
    new = metrics_view(new_metrics)
    current = metrics_view(current_metrics)
    return new["overkill_rate"] > current["overkill_rate"] + tolerance


# ---------------------------------------------------------------------------
# FunctionTool: commit the validated script to state before evaluation
# ---------------------------------------------------------------------------


def save_validated_ensemble_script_fn(tool_context, script: str) -> str:
    """
    Write the (possibly validator-corrected) script to state["ensemble_script"]
    so that evaluate_ensemble_fn always reads an authoritative value from state.
    Call this after code_validator_agent returns and before evaluate_ensemble_fn.
    """
    tool_context.state["ensemble_script"] = script
    return "Validated ensemble script committed to state['ensemble_script']."


_save_validated_script_tool = FunctionTool(func=save_validated_ensemble_script_fn)


def load_ensemble_script_fn(tool_context) -> str:
    """
    Return state["ensemble_script"] (the script written by ensemble_coder_agent)
    into THIS turn. Call this FIRST so the cache-check and validation steps have the
    script text without relying on conversation history (include_contents="none").
    """
    script = tool_context.state.get("ensemble_script", "")
    if not script:
        return "ERROR: ensemble_script not found in state."
    return script


_load_ensemble_script_tool = FunctionTool(func=load_ensemble_script_fn)

# ---------------------------------------------------------------------------
# Main evaluation FunctionTool
# ---------------------------------------------------------------------------


def evaluate_ensemble_fn(tool_context) -> str:
    """
    Read state["ensemble_script"], parse METRICS, update ensemble state, manage loop exit.

    Exit conditions (all via tool_context.actions.escalate inside this FunctionTool):

    1. Iteration cap reached (ensemble_iteration >= ENSEMBLE_LOOP_MAX after increment):
       Escalate — LoopAgent exits naturally after this iteration anyway, but we
       escalate early to be explicit.

    2. No improvement over ensemble_best_score:
       Escalate — no point running more iterations if this attempt failed to improve.
       The best ensemble result (if any) is already saved to ensemble.json.
    """
    # Read from state so the LLM never has to pass the full script as an argument
    script = tool_context.state.get("ensemble_script", "")
    n = int(tool_context.state.get("ensemble_iteration", 0))
    ensemble_best         = float(tool_context.state.get("ensemble_best_score",   0.0))
    ensemble_best_overkill = float(tool_context.state.get("ensemble_best_overkill", 1.0))
    ensemble_best_accuracy = float(tool_context.state.get("ensemble_best_accuracy", 0.0))
    ensemble_best_f1 = float(tool_context.state.get("ensemble_best_f1", 0.0))

    # ---- execute the script ----
    logger.info("Ensemble evaluator running script (iteration=%d)", n)
    result = code_runner.run_script(script, timeout=config.TIMEOUT_SECONDS)
    metrics = parse_metrics(result.stdout)
    # Persistence-boundary guard: a degenerate ensemble run must not be persisted
    # as a valid candidate (would poison ensemble selection + final reporting).
    metrics = metric_guard.guard_metrics(
        metrics, result.duration_ms, context=f"phase3 ensemble iteration {n}"
    )

    run_ok       = result.returncode == 0 and metrics is not None
    new_score    = float(metrics.ng_recall)     if metrics else 0.0
    new_overkill = float(metrics.overkill_rate) if metrics else 1.0

    # Parse ensemble diagnostic signals
    calibration_stats = parse_calibration_stats(result.stdout)
    epoch_logs = parse_epoch_logs(result.stdout)
    if calibration_stats:
        tool_context.state["latest_ensemble_calibration_stats"] = calibration_stats
    if epoch_logs:
        tool_context.state["latest_ensemble_epoch_logs"] = epoch_logs

    # ---- decide improvement (spec §8: P0/P1=ng_recall, P2=overkill constraint) ----
    current_metrics = {
        "accuracy": ensemble_best_accuracy,
        "ng_recall": ensemble_best,
        "miss_rate": max(0.0, 1.0 - ensemble_best),
        "overkill_rate": ensemble_best_overkill,
        "f1": ensemble_best_f1,
    }
    improved = run_ok and _is_ensemble_improvement(
        new_score,
        new_overkill,
        ensemble_best,
        ensemble_best_overkill,
        new_metrics=metrics,
        current_metrics=current_metrics,
    )
    failure_reason = None
    if run_ok and metrics is not None:
        if _has_overkill_regression(metrics, current_metrics):
            failure_reason = "overkill_regression"
        elif metrics.ng_recall >= 1.0 and metrics.overkill_rate >= 1.0:
            failure_reason = "degenerate_all_ng"
        elif not improved:
            failure_reason = "no_acceptance_improvement"
    elif not run_ok:
        failure_reason = "execution_failed"
    tool_context.state["latest_ensemble_failure_reason"] = failure_reason

    if improved:
        tool_context.state["ensemble_best_score"]   = new_score
        tool_context.state["ensemble_best_overkill"] = new_overkill
        tool_context.state["ensemble_best_accuracy"] = float(metrics.accuracy)
        tool_context.state["ensemble_best_f1"] = float(metrics.f1)
        _save_best_ensemble(
            n, script, new_score, new_overkill, metrics_to_dict(metrics),
            calibration_stats=tool_context.state.get("latest_ensemble_calibration_stats"),
        )
        logger.info(
            "New ensemble best: ng_recall=%.4f overkill=%.4f (was ng_recall=%.4f overkill=%.4f)",
            new_score, new_overkill, ensemble_best, ensemble_best_overkill,
        )
    else:
        logger.info(
            "No improvement (new_recall=%.4f new_overkill=%.4f, best_recall=%.4f best_overkill=%.4f, run_ok=%s)",
            new_score, new_overkill, ensemble_best, ensemble_best_overkill, run_ok,
        )

    # ---- increment iteration counter ----
    n_next = n + 1
    tool_context.state["ensemble_iteration"] = n_next

    # ---- persist per-attempt checkpoint ----
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    attempt_data = {
        "ensemble_iteration": n,
        "returncode":         result.returncode,
        "timed_out":          result.timed_out,
        "duration_ms":        round(result.duration_ms, 1),
        "improved":                improved,
        "new_score":               new_score,
        "new_overkill":            new_overkill,
        "failure_reason":          failure_reason,
        "ensemble_best_score":     tool_context.state["ensemble_best_score"],
        "ensemble_best_overkill":  tool_context.state.get("ensemble_best_overkill", 1.0),
        "metrics":            metrics_to_dict(metrics) if metrics else None,
        "calibration_stats":  tool_context.state.get("latest_ensemble_calibration_stats"),
        "stdout_tail":        result.stdout[-3000:],
        "stderr_tail":        result.stderr[-1000:],
    }
    save_checkpoint(config.ckpt_ensemble_attempt(n), attempt_data)
    _record_tried_ensemble_approach(tool_context.state, n, metrics, improved, run_ok, failure_reason)

    # Build human-readable metrics line
    if metrics:
        metrics_line = (
            f"ng_recall={metrics.ng_recall:.3f}  miss_rate={metrics.miss_rate:.3f}  "
            f"overkill={metrics.overkill_rate:.3f}  f1={metrics.f1:.3f}  "
            f"threshold={metrics.threshold}  "
            f"({'NEW BEST' if improved else 'no improvement'})"
        )
    else:
        metrics_line = (
            f"FAILED (rc={result.returncode}, timed_out={result.timed_out})  "
            f"stderr: {result.stderr[-200:]}"
        )

    summary_prefix = (
        f"[ensemble_iteration={n}]  {metrics_line}  |  "
        f"ensemble_best={tool_context.state['ensemble_best_score']:.4f}"
    )

    # ---- Fallback ensemble.json write ----
    # Guarantee ensemble.json exists at every exit point so submission_agent
    # never hits FileNotFoundError.  Written only when no prior iteration
    # already saved it (i.e. improved=True path wrote it above).
    def _ensure_ensemble_json() -> None:
        if not config.CKPT_ENSEMBLE.exists():
            fallback = tool_context.state.get("best_pipeline_script", "")
            _save_best_ensemble(
                n, fallback,
                tool_context.state.get("ensemble_best_score", 0.0),
                tool_context.state.get("ensemble_best_overkill", 1.0),
                None,
            )
            logger.info("Fallback ensemble.json written using best_pipeline_script.")

    # ---- Exit condition 0: token budget hard stop ----
    # Phase 3 has no other token brake and inherits Phase 2's cumulative token_count
    # within an attempt. Mirror the refinement evaluator's TOKEN_BUDGET stop so the
    # ensemble loop cannot run away over budget. The ensemble loop is a LoopAgent, so
    # escalate exits it cleanly; _ensure_ensemble_json guarantees Phase 4 has input.
    if tool_context.state.get("token_count", 0) >= config.TOKEN_BUDGET:
        _ensure_ensemble_json()
        tool_context.actions.escalate = True
        logger.warning(
            "Ensemble: token budget exhausted (%d >= %d) — escalating.",
            tool_context.state.get("token_count", 0), config.TOKEN_BUDGET,
        )
        return (
            f"{summary_prefix}\n"
            f"TOKEN_BUDGET_STOP: token_count >= TOKEN_BUDGET ({config.TOKEN_BUDGET}). "
            "Escalating ensemble loop; best ensemble preserved."
        )

    # ---- Exit condition 1: iteration cap ----
    if n_next >= config.ENSEMBLE_LOOP_MAX:
        _ensure_ensemble_json()
        tool_context.actions.escalate = True
        logger.info("Ensemble iteration cap reached (%d >= %d) — escalating.", n_next, config.ENSEMBLE_LOOP_MAX)
        return (
            f"{summary_prefix}\n"
            f"ENSEMBLE_CAP: iteration {n_next} >= ENSEMBLE_LOOP_MAX ({config.ENSEMBLE_LOOP_MAX}). "
            "Escalating ensemble loop."
        )

    # ---- Exit condition 2: no improvement cap ----
    # Let the loop run for at least 3 iterations (0, 1, 2) to explore the diverse
    # ensemble strategies (Weighted-average, Specialized Thresholds, Augmentation Diversity).
    # Only escalate if we have completed these baseline iterations and have seen
    # consecutive iterations with no improvement.
    no_improve_count = int(tool_context.state.get("ensemble_no_improve_count", 0))
    if not improved:
        no_improve_count += 1
        tool_context.state["ensemble_no_improve_count"] = no_improve_count
    else:
        tool_context.state["ensemble_no_improve_count"] = 0
        no_improve_count = 0

    if run_ok and not improved and n_next >= 3 and no_improve_count >= 2:
        _ensure_ensemble_json()
        tool_context.actions.escalate = True
        logger.info("Ensemble no-improvement cap reached — escalating.")
        return (
            f"{summary_prefix}\n"
            "NO_IMPROVEMENT: ensemble score did not improve and cap reached. "
            "Escalating ensemble loop."
        )

    # ---- Continue ----
    return (
        f"{summary_prefix}\n"
        f"CONTINUE: ensemble loop proceeding (ensemble_iteration now {n_next})."
    )


_evaluate_ensemble_tool = FunctionTool(func=evaluate_ensemble_fn)

# ---------------------------------------------------------------------------
# Agent instruction
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are the Ensemble Evaluator Agent for Phase 3.

Your role is to validate the ensemble script, execute it, parse results, and
manage ensemble loop termination.

---
## STEP 1 — Load the current ensemble script

Call `load_ensemble_script_fn` FIRST. It returns the script written by
`ensemble_coder_agent` in the previous step. Do NOT rely on conversation history
for the script text — use the value this tool returns for all subsequent steps.

---
## STEP 2 — Validate

First call `check_validation_cache_fn` with the script text returned by `load_ensemble_script_fn`.
- If it returns "CACHE_HIT: VALIDATED":
  The script was already validated by the coder agent — skip `code_validator_agent`.
  Call `save_validated_ensemble_script_fn` with the original script and proceed to STEP 3.
- If it returns "CACHE_HIT: VALIDATION_FAILED":
  The script is known-broken from the coder's validation pass — skip `code_validator_agent`.
  Call `save_validated_ensemble_script_fn` with the original script, then proceed to STEP 3.
  `evaluate_ensemble_fn` will run the script, detect returncode != 0, record no improvement,
  and advance the loop counters — this is the correct outcome for a broken script.
- If it returns "CACHE_MISS":
  Call `code_validator_agent` with the script text.
  - If it returns "VALIDATED_SCRIPT:": extract the corrected script that follows.
    Call `store_validation_cache_fn(corrected_script, "VALIDATED")`.
  - If it returns "VALIDATION_FAILED": use the original script unchanged.
    Call `store_validation_cache_fn(original_script, "VALIDATION_FAILED")`.
  Then call `save_validated_ensemble_script_fn` with the final script (corrected or original).

---
## STEP 3 — Evaluate and manage the ensemble loop

Call `evaluate_ensemble_fn` with no arguments — it reads the script directly
from `state["ensemble_script"]` (committed in the previous step).

This tool:
  - Runs the script and parses METRICS
  - Compares ng_recall against state["ensemble_best_score"]
  - Rejects overkill regressions even when recall improves
  - Updates state["ensemble_best_score"] and saves checkpoints/ensemble.json if improved
  - Increments state["ensemble_iteration"]
  - Saves checkpoints/ensemble_[N].json for this attempt
  - Escalates (exits the ensemble LoopAgent) when:
      * ensemble_iteration reaches ENSEMBLE_LOOP_MAX (3), OR
      * the script did not improve ensemble_best_score

---
## STEP 4 — Report

After `evaluate_ensemble_fn` returns, report:
1. The key metrics: ng_recall, miss_rate, overkill_rate, f1, threshold
2. Whether this attempt improved ensemble_best_score
3. If not improved, whether the failure was overkill_regression, degenerate_all_ng, or no_acceptance_improvement
4. Current ensemble_iteration and ensemble_best_score
5. The exit decision (CONTINUE / ENSEMBLE_CAP / NO_IMPROVEMENT) as returned by the tool

Do NOT call any other tools. Do NOT modify loop counters manually.
All loop state management is handled inside `evaluate_ensemble_fn`.
"""

# ---------------------------------------------------------------------------
# Ensemble evaluator agent
# ---------------------------------------------------------------------------

ensemble_evaluator_agent = LlmAgent(
    name="ensemble_evaluator_agent",
    model=config.MODEL,
    description=(
        "Validates and runs the ensemble script, updates ensemble_best_score on improvement, "
        "saves ensemble_{{N}}.json per attempt and ensemble.json for the best result. "
        "Escalates via FunctionTool when iteration cap or no-improvement conditions are met."
    ),
    instruction=_INSTRUCTION,
    tools=[_load_ensemble_script_tool, _evaluate_ensemble_tool, _save_validated_script_tool, code_validator_tool, check_validation_cache_tool, store_validation_cache_tool],
    include_contents="none",
    before_model_callback=log_context_size_callback,
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
