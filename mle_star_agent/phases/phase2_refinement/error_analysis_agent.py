import json as _json
import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.shared.callbacks import (
    budget_stop_callback,
    count_tokens_callback,
    log_context_size_callback,
    rate_limit_retry_callback,
)
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint, save_checkpoint
from mle_star_agent.shared.checkpoint_lineage import lineage_matches, stable_json_sha256

logger = logging.getLogger(__name__)


def _evaluated_inner_iteration(state: dict) -> int:
    return max(0, int(state.get("inner_iteration", 0)) - 1)


def _error_report_lineage(state: dict) -> dict:
    evidence = state.get("latest_error_analysis", {})
    return {
        "error_analysis_sha256": stable_json_sha256(evidence),
        "error_analysis_path": state.get("latest_error_analysis_path", ""),
    }


def _is_error_analysis_report_current(data: dict, state: dict) -> bool:
    return lineage_matches(data.get("lineage"), _error_report_lineage(state))


def _block_no_evidence(tool_context, reason: str) -> str:
    """
    Stop the inner loop after the single instrumentation-repair pass failed to
    produce usable per-sample evidence.
    """
    tool_context.state["error_analysis_blocked"] = True
    tool_context.actions.escalate = True
    logger.error("Gate: blocking blind refinement — %s", reason)
    return (
        f"BLOCK_NO_EVIDENCE: {reason} "
        "A prior instrumentation-repair iteration already ran without producing "
        "usable ERROR_ANALYSIS/PREDICTIONS evidence. Escalating before planner/coder "
        "to prevent blind refinement."
    )


def _allow_no_evidence(tool_context, reason: str) -> str:
    """
    Allow one iteration to proceed without per-sample evidence.
    Sets instrumentation_required so the coder prioritises adding PREDICTIONS output.
    A later missing-evidence gate blocks blind refinement.
    """
    if tool_context.state.get("error_analysis_repair_attempted"):
        return _block_no_evidence(tool_context, reason)
    tool_context.state["error_analysis_instrumentation_required"] = True
    tool_context.state["error_analysis_repair_attempted"] = True
    tool_context.state.pop("error_analysis_blocked", None)
    logger.warning("Gate: allowing iteration without evidence — %s", reason)
    return (
        f"ALLOW_NO_EVIDENCE: {reason} "
        "state['error_analysis_instrumentation_required'] = True. "
        "The coder MUST emit PREDICTIONS and ERROR_ANALYSIS output this iteration."
    )


def check_error_analysis_gate_fn(tool_context) -> str:
    """
    Guard against blind refinement after the first inner iteration.

    If per-sample evidence is missing, the gate allows one instrumentation-repair
    iteration, then blocks further blind refinement. The outer loop is not killed
    here; this guard only prevents planner/coder from choosing another strategy
    without evidence.
    """
    m = int(tool_context.state.get("inner_iteration", 0))
    if m <= 0:
        return "ALLOW: first refinement iteration can proceed without prior error-analysis report."

    report = tool_context.state.get("error_analysis_report")

    # On resume, state["error_analysis_report"] is absent even if the checkpoint was
    # written by the previous session. Load it from disk before deciding.
    if not isinstance(report, dict):
        n = int(tool_context.state.get("outer_iteration", 0))
        ckpt_path = config.ckpt_error_analysis_report(n, m - 1)
        if checkpoint_exists(ckpt_path):
            data = load_checkpoint(ckpt_path)
            report = data.get("error_analysis_report")
            if isinstance(report, dict):
                tool_context.state["error_analysis_report"] = report

    if not isinstance(report, dict):
        return _allow_no_evidence(
            tool_context,
            "no error_analysis_report found for the previous iteration.",
        )

    if report.get("evidence_available") is not True:
        return _allow_no_evidence(
            tool_context,
            "error_analysis_report exists but evidence_available is not True.",
        )

    consistency = report.get("metrics_consistency") or {}
    if consistency.get("matches_metrics") is False:
        logger.warning(
            "Gate: metrics consistency check failed — allowing iteration with warning. "
            "fp=%s fn=%s", report.get("fp_count"), report.get("fn_count"),
        )
        return (
            "ALLOW_CONSISTENCY_WARNING: error_analysis_report metrics consistency failed. "
            "Proceeding — treat FP/FN counts as approximate."
        )

    tool_context.state.pop("error_analysis_instrumentation_required", None)
    tool_context.state.pop("error_analysis_repair_attempted", None)
    tool_context.state.pop("error_analysis_blocked", None)
    return "ALLOW: valid error_analysis_report present for next refinement."


def _capped_evidence_block(evidence: dict) -> str:
    """
    Render the per-sample error-analysis evidence into the LLM turn, capping the
    FP/FN sample arrays to ERROR_ANALYSIS_SAMPLE_CAP each. The full set stays in
    state["latest_error_analysis"] and on disk — only the prompt view is capped.
    This is what surfaces fp_samples / fn_samples / probability_summary /
    metrics_consistency under include_contents="none".
    """
    cap = config.ERROR_ANALYSIS_SAMPLE_CAP
    fp_samples = list(evidence.get("fp_samples", []) or [])
    fn_samples = list(evidence.get("fn_samples", []) or [])
    view = {
        "available": evidence.get("available"),
        "fp_count": evidence.get("fp_count"),
        "fn_count": evidence.get("fn_count"),
        "fp_samples": fp_samples[:cap],
        "fn_samples": fn_samples[:cap],
        "fp_samples_truncated": len(fp_samples) > cap,
        "fn_samples_truncated": len(fn_samples) > cap,
        "probability_summary": evidence.get("probability_summary", {}),
        "per_lot": evidence.get("per_lot", {}),
        "metrics_consistency": evidence.get("metrics_consistency", {}),
    }
    return _json.dumps(view, default=str)


def load_latest_error_analysis_fn(tool_context) -> str:
    """
    Load the latest deterministic error-analysis evidence for the refinement
    attempt that just ran. Prefer state, then fall back to the checkpoint.

    Returns the per-sample evidence (fp_samples/fn_samples capped, plus
    probability_summary and metrics_consistency) directly in the tool output so
    the agent can interpret it under include_contents="none" — do NOT rely on
    reading state["latest_error_analysis"] from conversation history.
    """
    n = int(tool_context.state.get("outer_iteration", 0))
    m = _evaluated_inner_iteration(tool_context.state)

    state_evidence = tool_context.state.get("latest_error_analysis")
    if isinstance(state_evidence, dict):
        return (
            "ERROR_ANALYSIS_LOADED_FROM_STATE:\n"
            + _capped_evidence_block(state_evidence)
        )

    ckpt_path = config.ckpt_error_analysis(n, m)
    if not checkpoint_exists(ckpt_path):
        return (
            f"ERROR_ANALYSIS_MISSING: no checkpoint at {ckpt_path}. "
            "Do not infer per-sample patterns from aggregate metrics alone."
        )

    evidence = load_checkpoint(ckpt_path)
    tool_context.state["latest_error_analysis"] = evidence
    tool_context.state["latest_error_analysis_path"] = str(ckpt_path)
    return (
        f"ERROR_ANALYSIS_LOADED ({ckpt_path.name}):\n"
        + _capped_evidence_block(evidence)
    )


def check_error_analysis_report_checkpoint_fn(tool_context) -> str:
    """
    If the interpretation report for the latest error-analysis evidence exists,
    load it into state and skip re-analysis.
    """
    n = int(tool_context.state.get("outer_iteration", 0))
    m = _evaluated_inner_iteration(tool_context.state)
    ckpt_path = config.ckpt_error_analysis_report(n, m)

    if not checkpoint_exists(ckpt_path):
        return f"CHECKPOINT_NOT_FOUND: no error_analysis_report_{n}_{m}.json."

    data = load_checkpoint(ckpt_path)
    if not _is_error_analysis_report_current(data, tool_context.state):
        return (
            f"CHECKPOINT_STALE: error_analysis_report_{n}_{m}.json does not match "
            "the latest deterministic error-analysis evidence."
        )

    report = data.get("error_analysis_report", {})
    tool_context.state["error_analysis_report"] = report
    return (
        f"CHECKPOINT_FOUND: error_analysis_report_{n}_{m}.json loaded. "
        f"Dominant failure: {report.get('dominant_failure', 'unknown')}."
    )


def write_error_analysis_report_fn(
    tool_context,
    dominant_failure: str,
    threshold_fix_possible: bool,
    evidence_summary: str,
    recommended_target_component: str,
    recommended_changes: str,
) -> str:
    """
    Persist the LLM interpretation of deterministic FP/FN evidence.

    recommended_changes may be a JSON list string or plain text.
    """
    n = int(tool_context.state.get("outer_iteration", 0))
    m = _evaluated_inner_iteration(tool_context.state)
    evidence = tool_context.state.get("latest_error_analysis", {})

    try:
        changes = _json.loads(recommended_changes)
    except Exception:
        changes = recommended_changes

    report = {
        "outer_iteration": n,
        "inner_iteration": m,
        "dominant_failure": dominant_failure,
        "threshold_fix_possible": bool(threshold_fix_possible),
        "evidence_summary": evidence_summary,
        "recommended_target_component": recommended_target_component,
        "recommended_changes": changes,
        "evidence_available": evidence.get("available"),
        "fp_count": evidence.get("fp_count"),
        "fn_count": evidence.get("fn_count"),
        "metrics_consistency": evidence.get("metrics_consistency"),
        "source_error_analysis_path": tool_context.state.get("latest_error_analysis_path", ""),
    }

    tool_context.state["error_analysis_report"] = report

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.ckpt_error_analysis_report(n, m)
    save_checkpoint(ckpt_path, {
        "lineage": _error_report_lineage(tool_context.state),
        "error_analysis_report": report,
    })

    logger.info(
        "Error-analysis report written for outer=%d inner=%d dominant_failure=%s",
        n,
        m,
        dominant_failure,
    )
    return (
        f"Error-analysis report saved to {ckpt_path.name}. "
        f"Recommended target: '{recommended_target_component}'."
    )


_load_latest_tool = FunctionTool(func=load_latest_error_analysis_fn)
_check_report_ckpt_tool = FunctionTool(func=check_error_analysis_report_checkpoint_fn)
_write_report_tool = FunctionTool(func=write_error_analysis_report_fn)
_check_gate_tool = FunctionTool(func=check_error_analysis_gate_fn)


_GATE_INSTRUCTION = """You are the Error Analysis Gate for Phase 2 Refinement.

Call `check_error_analysis_gate_fn` immediately.

Your only job is to check whether per-sample evidence is available:
- ALLOW: valid evidence present — proceed normally
- ALLOW_NO_EVIDENCE: evidence missing for the first time — state["error_analysis_instrumentation_required"]
  has been set to True; the coder will add PREDICTIONS output this iteration
- BLOCK_NO_EVIDENCE: evidence is still missing after the repair pass; the gate
  escalated before planner/coder to stop blind refinement
- ALLOW_CONSISTENCY_WARNING: evidence present but metrics may be approximate

Do NOT set stop_outer_loop.
Do not call any other tools.
"""


_INSTRUCTION = """You are the Error Analysis Agent for Phase 2 Refinement.

Your role is to interpret deterministic per-sample FP/FN evidence after each
refinement evaluation, then write a compact report that the next refinement
iteration can use.

---
## STEP 1 - Load deterministic evidence

Call `load_latest_error_analysis_fn` immediately.
- If it returns ERROR_ANALYSIS_MISSING: report that no per-sample evidence exists
  and stop. Do not infer sample-level patterns from aggregate metrics alone.
- Otherwise continue.

---
## STEP 2 - Check for existing report checkpoint

Call `check_error_analysis_report_checkpoint_fn`.
- If it returns CHECKPOINT_FOUND: report the loaded dominant failure and stop.
- If it returns CHECKPOINT_NOT_FOUND or CHECKPOINT_STALE: continue.

---
## STEP 3 - Interpret the evidence

Use the JSON block returned by `load_latest_error_analysis_fn` in STEP 1 (do NOT
rely on conversation history). Focus on:
- `available`
- `fp_count`, `fn_count`
- `fp_samples`, `fn_samples` (capped to the most recent samples; `*_truncated`
  flags indicate when the full set on disk is larger)
- `probability_summary`
- `metrics_consistency`

If `available=false`, the dominant failure is "missing_evidence"; recommend that
the next refinement script emit PREDICTIONS / ERROR_ANALYSIS with sample IDs,
true label, predicted label, NG probability, and threshold.

If evidence is available:
- If FP count is far above the overkill budget, dominant_failure should usually
  be "overkill" or "g_false_positive_control".
- If FN count is nonzero while FP is already near budget, dominant_failure should
  usually be "missed_ng".
- If G and NG probability summaries overlap heavily, say threshold-only fixes are
  unlikely and recommend separability improvements.
- A high-recall/high-overkill model is not progress. Prefer recommendations that
  reduce FP while preserving or moving toward FN=0.

---
## STEP 4 - Write the report

Call `write_error_analysis_report_fn` with:
1. `dominant_failure`
2. `threshold_fix_possible`
3. `evidence_summary`
4. `recommended_target_component`
5. `recommended_changes` as a JSON list string when possible

Use one of these target components when appropriate:
- "g_false_positive_control"
- "threshold_sweep"
- "calibration"
- "preprocessing_normalization"
- "stereo_fusion"
- "model_architecture"
- "training_schedule"
- "error_analysis_instrumentation"

---
## STEP 5 - Summarise

Briefly report:
- dominant failure
- whether threshold-only adjustment is likely sufficient
- recommended next target
"""


error_analysis_gate_agent = LlmAgent(
    name="error_analysis_gate_agent",
    model=config.MODEL,
    description=(
        "Pre-refinement gate that blocks repeated refinement attempts when the "
        "previous evaluation lacks a valid per-sample error-analysis report."
    ),
    instruction=_GATE_INSTRUCTION,
    tools=[_check_gate_tool],
    include_contents="none",
    before_model_callback=log_context_size_callback,
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)


error_analysis_agent = LlmAgent(
    name="error_analysis_agent",
    model=config.MODEL_PRO,
    description=(
        "Interprets deterministic FP/FN error-analysis checkpoints after each "
        "refinement evaluation and writes state['error_analysis_report'] for the "
        "next refinement iteration."
    ),
    instruction=_INSTRUCTION,
    tools=[_load_latest_tool, _check_report_ckpt_tool, _write_report_tool],
    include_contents="none",
    before_model_callback=[log_context_size_callback, budget_stop_callback],
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
