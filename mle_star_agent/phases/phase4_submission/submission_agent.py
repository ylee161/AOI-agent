import json
import logging
import re
from typing import Any, Optional

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.guards.code_validator_agent import code_validator_tool
from mle_star_agent.shared import code_runner, metric_guard
from mle_star_agent.shared.callbacks import count_tokens_callback, rate_limit_retry_callback
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.checkpoint_lineage import lineage_matches, script_lineage
from mle_star_agent.shared.metrics_parser import metrics_to_dict, parse_metrics
from mle_star_agent.shared.diagnosis_scorer import parse_calibration_stats

logger = logging.getLogger(__name__)


def _extract_json_block(stdout: str, marker: str) -> Optional[dict]:
    marker_match = re.search(rf"{re.escape(marker)}:\s*", stdout)
    if not marker_match:
        return None
    start = marker_match.end()
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(stdout[start:].lstrip())
    except json.JSONDecodeError:
        logger.warning("Could not parse %s JSON block from stdout.", marker)
        return None
    if not isinstance(payload, dict):
        logger.warning("%s JSON block must be an object.", marker)
        return None
    return payload


def _load_test_sample_ids() -> list[str]:
    if not checkpoint_exists(config.CKPT_DATA_SPLIT):
        return []
    data_split = load_checkpoint(config.CKPT_DATA_SPLIT)
    return [s.get("sample_id", "") for s in data_split.get("test", []) if s.get("sample_id")]


def _acceptance(metrics_dict: Optional[dict], threshold: Any) -> dict:
    if not metrics_dict:
        return {
            "relaxed_minimum_pass": False,
            "final_target_pass": False,
            "reasons": ["metrics_missing"],
        }

    reasons = []
    checks = {
        "accuracy":          metrics_dict.get("accuracy",      0.0) >= config.ACCURACY_RELAXED_MIN,
        "ng_recall":         metrics_dict.get("ng_recall",     0.0) >= config.NG_RECALL_RELAXED_MIN,
        "miss_rate":         metrics_dict.get("miss_rate",     1.0) <= config.MISS_RATE_RELAXED_MAX,
        "overkill_rate":     metrics_dict.get("overkill_rate", 1.0) <= config.OVERKILL_RELAXED_MAX,
        "threshold_recorded": threshold is not None,
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)

    final_checks = {
        "ng_recall":     metrics_dict.get("ng_recall",     0.0) >= config.NG_RECALL_FINAL_MIN,
        "miss_rate":     metrics_dict.get("miss_rate",     1.0) <= config.MISS_RATE_FINAL_MAX,
        "overkill_rate": metrics_dict.get("overkill_rate", 1.0) <= config.OVERKILL_FINAL_MAX,
        "accuracy":      metrics_dict.get("accuracy",      0.0) >= config.ACCURACY_FINAL_MIN,
    }

    return {
        "relaxed_minimum_pass": all(checks.values()),
        "final_target_pass": all(final_checks.values()),
        "checks": checks,
        "final_checks": final_checks,
        "reasons": reasons,
    }


def _select_submission_script_from_sources(state) -> tuple[str, str]:
    script = ""
    source = ""

    if checkpoint_exists(config.CKPT_ENSEMBLE):
        data = load_checkpoint(config.CKPT_ENSEMBLE)
        script = data.get("ensemble_script", "")
        if script:
            source = "ensemble_checkpoint"

    if not script:
        script = state.get("ensemble_script", "")
        if script:
            source = "ensemble_state"

    if not script and checkpoint_exists(config.CKPT_BEST_PIPELINE):
        data = load_checkpoint(config.CKPT_BEST_PIPELINE)
        script = data.get("best_pipeline_script", "")
        if script:
            source = "best_pipeline_checkpoint"

    if not script:
        script = state.get("best_pipeline_script", "")
        if script:
            source = "best_pipeline_state"

    return script, source


def _submission_lineage(script: str) -> dict:
    return script_lineage(script, script_key="submission_script_sha256")


def _is_submission_checkpoint_current(data: dict, selected_script: str) -> bool:
    return lineage_matches(data.get("lineage"), _submission_lineage(selected_script))


def check_submission_checkpoint_fn(tool_context) -> str:
    """Restore submission.json if it already exists and skip final execution."""
    if not checkpoint_exists(config.CKPT_SUBMISSION):
        return "CHECKPOINT_NOT_FOUND: proceed to final submission."

    data = load_checkpoint(config.CKPT_SUBMISSION)
    selected_script, source = _select_submission_script_from_sources(tool_context.state)
    if not selected_script:
        return (
            "CHECKPOINT_NOT_FOUND: no script available to verify lineage — "
            "cannot confirm submission.json is current. Proceeding to final submission."
        )
    if not _is_submission_checkpoint_current(data, selected_script):
        return (
            "CHECKPOINT_STALE: submission.json lineage does not match the currently "
            f"selected script from {source}, data split, or acceptance config. "
            "Proceeding to final submission."
        )

    tool_context.state["final_metrics"] = data.get("metrics", {})
    tool_context.state["submission_result"] = data
    return (
        "CHECKPOINT_FOUND: submission.json loaded. "
        f"relaxed_minimum_pass={data.get('pass_fail', {}).get('relaxed_minimum_pass')}"
    )


def select_submission_script_fn(tool_context) -> str:
    """
    Select the script for final submission. Prefer the persisted best ensemble,
    then live ensemble state, then persisted best pipeline, then live best pipeline.
    """
    script, source = _select_submission_script_from_sources(tool_context.state)

    if not script:
        return (
            "ERROR: no submission script found in ensemble checkpoint/state or "
            "best pipeline checkpoint/state."
        )

    tool_context.state["submission_script"] = script
    tool_context.state["submission_source"] = source

    # Surface modality so any validator-driven edits stay modality-correct. Default
    # "stereo" preserves existing behaviour; mono notes the single-image contract.
    input_modality = tool_context.state.get("input_modality", "stereo")
    if input_modality == "mono":
        modality_block = (
            "INPUT_MODALITY: mono — single image per sample (`img`); no `img_l`/`img_r` "
            "pair. Do not introduce stereo loading or stereo-fusion when editing the script."
        )
    else:
        modality_block = "INPUT_MODALITY: stereo"

    return (
        f"{modality_block}\n\n"
        f"Selected submission script from {source}.\n\nSUBMISSION_SCRIPT:\n{script}"
    )


def save_validated_submission_script_fn(tool_context, script: str) -> str:
    """Commit the validator-corrected final script before execution."""
    tool_context.state["submission_script"] = script
    return "Validated submission script committed to state['submission_script']."


def evaluate_submission_fn(tool_context, script: str) -> str:
    """
    Execute final script, parse metrics and optional SUBMISSION_JSON payload, write
    state["final_metrics"], and save checkpoints/submission.json.
    """
    logger.info("Running final submission script from %s", tool_context.state.get("submission_source"))
    result = code_runner.run_script(script, timeout=config.TIMEOUT_SECONDS)
    metrics = parse_metrics(result.stdout)
    # Persistence-boundary guard: never certify a degenerate run as the final submission.
    metrics = metric_guard.guard_metrics(
        metrics, result.duration_ms, context="phase4 submission"
    )
    metrics_dict = metrics_to_dict(metrics) if metrics else None

    payload = _extract_json_block(result.stdout, "SUBMISSION_JSON") or {}
    predictions = payload.get("predictions", [])
    error_samples = payload.get("error_samples", [])
    ambiguous_samples = payload.get("ambiguous_samples", [])
    threshold = (metrics_dict or {}).get("threshold")
    pass_fail = _acceptance(metrics_dict, threshold)

    submission = {
        "script_source": tool_context.state.get("submission_source", "unknown"),
        "lineage": _submission_lineage(script),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_ms": round(result.duration_ms, 1),
        "metrics": metrics_dict,
        "calibration_stats": parse_calibration_stats(result.stdout),
        "pass_fail": pass_fail,
        "predictions": predictions,
        "error_samples": error_samples,
        "ambiguous_samples": ambiguous_samples,
        "test_sample_ids": payload.get("test_sample_ids") or _load_test_sample_ids(),
        "model_version": payload.get("model_version", "submission_agent_final"),
        "dataset_version": payload.get("dataset_version", config.CKPT_DATA_SPLIT.name),
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-1000:],
    }

    tool_context.state["final_metrics"] = metrics_dict or {}
    tool_context.state["submission_result"] = submission
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(config.CKPT_SUBMISSION, submission)

    if metrics:
        status = "PASS" if pass_fail["relaxed_minimum_pass"] else "FAIL"
        return (
            f"Submission {status}: ng_recall={metrics.ng_recall:.3f}, "
            f"miss_rate={metrics.miss_rate:.3f}, overkill={metrics.overkill_rate:.3f}, "
            f"accuracy={metrics.accuracy:.3f}, threshold={metrics.threshold}. "
            "Saved submission.json."
        )

    return (
        f"Submission FAILED: script did not produce METRICS (rc={result.returncode}, "
        f"timed_out={result.timed_out}). Saved submission.json."
    )


_check_checkpoint_tool = FunctionTool(func=check_submission_checkpoint_fn)
_select_script_tool = FunctionTool(func=select_submission_script_fn)
_save_validated_script_tool = FunctionTool(func=save_validated_submission_script_fn)
_evaluate_submission_tool = FunctionTool(func=evaluate_submission_fn)


_INSTRUCTION = """You are the Phase 4 Submission Agent.

Your role is to run the final AOI model script, write final metrics to
state["final_metrics"], and save checkpoints/submission.json.

---
## STEP 1 — Check checkpoint

Call `check_submission_checkpoint_fn` immediately.
- If it returns CHECKPOINT_FOUND: report the loaded result and stop.
- If it returns CHECKPOINT_NOT_FOUND: proceed to STEP 2.

---
## STEP 2 — Select final script

Call `select_submission_script_fn`.
It selects the best available script in this order:
1. checkpoints/ensemble.json
2. state["ensemble_script"]
3. checkpoints/best_pipeline.json
4. state["best_pipeline_script"]

If it returns ERROR, report the error and stop.

---
## STEP 3 — Ensure final-output contract

Use the SUBMISSION_SCRIPT text returned by `select_submission_script_fn`.
If needed, adapt the script so its final run:
- reads test sample IDs only from __DATA_SPLIT_PATH__
- evaluates only the test split for final metrics
- prints METRICS: {...} with the standard AOI metric keys
- preferably also prints SUBMISSION_JSON: {...} containing:
  predictions, error_samples, ambiguous_samples, test_sample_ids, model_version,
  and dataset_version
- every prediction object must include sample_id, prediction, confidence,
  ng_probability, threshold, and decision

Do not tune thresholds on the test split. Use the script's existing validation-derived threshold.

---
## STEP 4 — Validate

Call `code_validator_agent` with the final script text.
- If it returns VALIDATED_SCRIPT:, extract the corrected script.
- If it returns VALIDATION_FAILED, use the original selected script unchanged.

Then call `save_validated_submission_script_fn` with the final script text.

---
## STEP 5 — Execute and save

Call `evaluate_submission_fn` with the same final script text. This tool runs the
script, parses METRICS, writes state["final_metrics"], and saves
checkpoints/submission.json.

---
## STEP 6 — Report

Report:
1. Whether relaxed minimum acceptance passed
2. NG recall, miss rate, overkill rate, accuracy, threshold
3. The number of predictions, error samples, and ambiguous samples written
4. Confirmation that submission.json was saved

Do not manually write state or checkpoints outside the provided FunctionTools.
"""

# Inject the real data-split checkpoint path (config-driven, not a hardcoded relative
# path). Mirrors the baseline_coder / ensemble_coder placeholder pattern.
_INSTRUCTION = _INSTRUCTION.replace("__DATA_SPLIT_PATH__", str(config.CKPT_DATA_SPLIT))


submission_agent = LlmAgent(
    name="submission_agent",
    model=config.MODEL,
    description=(
        "Runs the final best ensemble or pipeline script on the AOI test split, "
        "writes state['final_metrics'], and saves checkpoints/submission.json."
    ),
    instruction=_INSTRUCTION,
    tools=[
        _check_checkpoint_tool,
        _select_script_tool,
        _save_validated_script_tool,
        _evaluate_submission_tool,
        code_validator_tool,
    ],
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
