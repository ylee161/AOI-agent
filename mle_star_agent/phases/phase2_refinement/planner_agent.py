import json
import logging
import re
from pathlib import Path

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
from mle_star_agent.shared.small_data_strategy_validator import KNOWN_FAILED_STRATEGY_FINGERPRINTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FunctionTool — restore tried_approaches from checkpoint
# ---------------------------------------------------------------------------

_REFINEMENT_CKPT_RE = re.compile(r"^refinement_(\d+)_(\d+)\.json$")


def _refinement_sort_key(path: Path) -> tuple:
    match = _REFINEMENT_CKPT_RE.match(path.name)
    if not match:
        return (999999, 999999, str(path))
    return (int(match.group(1)), int(match.group(2)), str(path))


def _iter_refinement_checkpoint_paths() -> list[Path]:
    """Return active and archived refinement attempt checkpoints."""
    candidates: list[Path] = []
    checkpoint_dir = config.CHECKPOINT_DIR
    if checkpoint_dir.exists():
        candidates.extend(checkpoint_dir.glob("refinement_*_*.json"))
        candidates.extend(checkpoint_dir.glob("archive_*/refinement_*_*.json"))
        candidates.extend(checkpoint_dir.glob("retry_archives/**/refinement_*_*.json"))

    unique = {
        p.resolve(): p
        for p in candidates
        if _REFINEMENT_CKPT_RE.match(p.name)
    }
    return sorted(unique.values(), key=_refinement_sort_key)


def _safe_load_checkpoint(path: Path) -> dict | None:
    try:
        return load_checkpoint(path)
    except RuntimeError as exc:
        logger.warning("Skipping unreadable refinement history checkpoint %s: %s", path, exc)
        return None


def _paired_pending_checkpoint(path: Path) -> dict:
    pending_path = path.with_name(path.stem + "_pending.json")
    if not pending_path.is_file():
        return {}
    return _safe_load_checkpoint(pending_path) or {}


def _infer_mechanism_class(target_component: str, strategy_text: str) -> str:
    text = f"{target_component} {strategy_text}".lower()
    if "fp_weight" in text or "false_positive" in text or "asymmetric" in text:
        return "fp_penalty_loss"
    if "focal" in text:
        return "focal_loss"
    if "sampler" in text or "oversampling" in text:
        return "class_balanced_sampler"
    if "temperature" in text or "platt" in text:
        return "temperature_scaling"
    if "isotonic" in text:
        return "isotonic_calibration"
    if "threshold" in text:
        return "threshold_acceptance_distance"
    if "stereo" in text or "diff" in text or "l/r" in text:
        return "stereo_diff_features"
    if "lot" in text or "normalization" in text:
        return "lot_normalization"
    return "unknown"


def _attempt_key(entry: dict) -> tuple:
    return (
        int(entry.get("outer", -1) or -1),
        int(entry.get("inner", -1) or -1),
        (entry.get("selected_strategy") or "").strip().lower(),
    )


def _attempt_from_refinement_checkpoint(path: Path) -> dict | None:
    data = _safe_load_checkpoint(path)
    if not isinstance(data, dict):
        return None

    metrics = data.get("metrics") or {}
    pending = _paired_pending_checkpoint(path)
    diagnosis = data.get("diagnosis_evidence") or {}
    plan = data.get("refinement_plan") or {}
    filename_match = _REFINEMENT_CKPT_RE.match(path.name)
    outer_from_name = int(filename_match.group(1)) if filename_match else 0
    inner_from_name = int(filename_match.group(2)) if filename_match else 0

    target_component = (
        diagnosis.get("target_component")
        or plan.get("target_component")
        or data.get("target_component")
        or "unknown"
    )
    selected_strategy = (
        diagnosis.get("strategy")
        or pending.get("strategy_executed")
        or data.get("selected_strategy")
        or ""
    )
    changes = plan.get("changes_summary") or "; ".join(pending.get("changes_applied", []) or [])
    mechanism = _infer_mechanism_class(target_component, selected_strategy or changes)
    improved = bool(data.get("improved", False))
    failure_reason = (
        data.get("failure_reason")
        or ("accepted" if improved else "execution_failed" if data.get("returncode", 0) != 0 else "no_improvement")
    )

    return {
        "outer": int(data.get("outer_iteration", outer_from_name) or 0),
        "inner": int(data.get("inner_iteration", inner_from_name) or inner_from_name),
        "target_component": target_component,
        "changes_summary": changes,
        "selected_strategy": selected_strategy,
        "strategy_fingerprint": {
            "target_component": target_component.strip().lower(),
            "mechanism_class": mechanism,
        },
        "result": {
            "ng_recall": round(float(metrics.get("ng_recall", data.get("new_score", 0.0)) or 0.0), 4),
            "miss_rate": round(float(metrics.get("miss_rate", 1.0) or 1.0), 4),
            "overkill": round(float(metrics.get("overkill_rate", data.get("new_overkill", 1.0)) or 1.0), 4),
            "accuracy": round(float(metrics.get("accuracy", 0.0) or 0.0), 4),
            "improved": improved,
        },
        "prediction_verification": data.get("prediction_verification"),
        "failure_reason": failure_reason,
        "source": "refinement_checkpoint",
        "source_path": str(path),
    }


def _recover_refinement_history() -> list[dict]:
    recovered = []
    for path in _iter_refinement_checkpoint_paths():
        entry = _attempt_from_refinement_checkpoint(path)
        if entry:
            recovered.append(entry)
    return recovered


def _merge_tried_histories(*histories: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen = set()
    for history in histories:
        for entry in history or []:
            if not isinstance(entry, dict):
                continue
            key = _attempt_key(entry)
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    merged.sort(key=lambda e: (
        int(e.get("outer", 0) or 0),
        int(e.get("inner", 0) or 0),
        e.get("source") == "refinement_checkpoint",
    ))
    return merged

def load_tried_approaches_fn(tool_context) -> str:
    """
    Restore tried_approaches from state, persisted history, and refinement
    checkpoints. Call this first so the planner has full history even after a
    session restart, RetryLoopAgent reset, or checkpoint archive migration.
    """
    # Strategy selection is per inner iteration. Clear any prior selection so the
    # gate cannot reuse a stale strategy after evaluation/error analysis advances.
    for key in (
        "refinement_strategy_candidates",
        "selected_refinement_strategy",
        "selected_refinement_strategy_key",
        "selected_refinement_strategy_outer",
        "selected_refinement_strategy_inner",
        "strategy_selection_reason",
    ):
        tool_context.state.pop(key, None)

    existing = list(tool_context.state.get("tried_approaches", []) or [])
    checkpoint_history = []
    if checkpoint_exists(config.CKPT_TRIED_APPROACHES):
        data = load_checkpoint(config.CKPT_TRIED_APPROACHES)
        checkpoint_history = data.get("tried_approaches", [])

    base_history = _merge_tried_histories(existing, checkpoint_history)
    tried = _merge_tried_histories(base_history, _recover_refinement_history())
    recovered_added = len(tried) - len(base_history)

    if tried:
        tool_context.state["tried_approaches"] = tried
        save_checkpoint(config.CKPT_TRIED_APPROACHES, {"tried_approaches": tried})
        logger.info(
            "Planner restored %d tried_approaches (%d recovered from refinement checkpoints).",
            len(tried),
            recovered_added,
        )
        if recovered_added:
            status = (
                f"RECOVERED_FROM_REFINEMENT_CHECKPOINTS: {recovered_added} missing "
                f"attempt(s) recovered; {len(tried)} total prior attempt(s) loaded."
            )
        elif existing:
            status = f"LOADED_FROM_STATE: {len(tried)} prior attempt(s) already in state."
        else:
            status = f"LOADED_FROM_CHECKPOINT: {len(tried)} prior attempt(s) restored."
    elif not checkpoint_exists(config.CKPT_TRIED_APPROACHES):
        status = "NO_HISTORY: tried_approaches.json does not exist — this is the first attempt."
    else:
        status = "NO_HISTORY: tried_approaches.json exists but is empty."

    # Modality-conditional guidance. Default "stereo" preserves all existing behaviour
    # (the static _INSTRUCTION already lists the stereo strategy options + fingerprints).
    # For mono we inject an authoritative override: the stereo failed-fingerprints are
    # NOT failures here (so they don't block valid mono approaches), and mono-appropriate
    # strategy options replace the stereo ones.
    input_modality = tool_context.state.get("input_modality", "stereo")
    if input_modality == "mono":
        modality_block = (
            "INPUT_MODALITY: mono\n"
            "MODALITY OVERRIDE (authoritative — overrides any stereo wording in your instructions):\n"
            "- This dataset has a SINGLE image per sample (`img`); there is no `img_l`/`img_r` pair.\n"
            "- The fingerprints (`stereo_fusion`, `global_feature_difference_only`) and "
            "(`stereo_fusion`, `two_independent_backbone_stereo`) are NOT failures for mono — "
            "they are irrelevant. Do NOT treat them as blocked, and do NOT propose stereo_fusion "
            "targets at all.\n"
            "- Present mono-appropriate strategy options instead: spatial attention on backbone "
            "features, mixup/CutMix augmentation, test-time augmentation (TTA), patch dropout, "
            "partial-unfreeze + small regularized head."
        )
    else:
        modality_block = "INPUT_MODALITY: stereo"

    return (
        f"{modality_block}\n\n"
        f"{status}\n\n"
        f"{_tried_approaches_view(tried)}\n\n"
        f"{_context_reports_block(tool_context.state)}"
    )


def _tried_approaches_view(tried: list) -> str:
    """
    Summarized tried_approaches for the planner PROMPT: the last K full entries plus
    an aggregate. The full list stays in state["tried_approaches"] — the rotation-lock
    helpers (_failed_attempt_count_for_target, _failed_strategy_fingerprint_keys) read
    that full list, NOT this view, so dedup/rotation accounting is unaffected (§6.4).
    """
    entries = [e for e in (tried or []) if isinstance(e, dict)]
    recent = entries[-config.TRIED_APPROACHES_RECENT_K:]

    failed_fingerprints: list[dict] = []
    reason_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for e in entries:
        result = e.get("result") or {}
        reason = (e.get("failure_reason") or "").strip().lower()
        if result.get("improved") is False:
            fingerprint = e.get("strategy_fingerprint") or {}
            if fingerprint:
                failed_fingerprints.append({
                    "target_component": fingerprint.get("target_component"),
                    "mechanism_class": fingerprint.get("mechanism_class"),
                })
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        # Rotation-lock-relevant: only true P0/P1 failures count toward a target.
        if _is_priority_correct_failure(e):
            tgt = (
                e.get("target_component")
                or (e.get("strategy_fingerprint") or {}).get("target_component")
                or "unknown"
            )
            tgt = str(tgt).strip().lower()
            target_counts[tgt] = target_counts.get(tgt, 0) + 1

    aggregate = {
        "total_attempts": len(entries),
        "failed_fingerprints": failed_fingerprints,
        "failure_counts_by_reason": reason_counts,
        "failed_attempts_by_target": target_counts,
    }
    return (
        f"TRIED_APPROACHES_RECENT (last {len(recent)} of {len(entries)}): "
        f"{json.dumps(recent, default=str)}\n"
        f"TRIED_APPROACHES_AGGREGATE: {json.dumps(aggregate, default=str)}"
    )


def _population_summary(population: list) -> list[dict]:
    """Compact view of refinement_population — metrics only, no full scripts."""
    summary = []
    for entry in population or []:
        if not isinstance(entry, dict):
            continue
        summary.append({
            "script_sha256": (entry.get("script_sha256", "") or "")[:12],
            "metrics": entry.get("metrics", {}),
            "archive_reason": entry.get("archive_reason", ""),
            "outer": entry.get("outer"),
            "inner": entry.get("inner"),
        })
    return summary


def _context_reports_block(state: dict) -> str:
    """
    Compact diagnosis + error-analysis + population block so the planner gets STEP 2
    context from this tool call rather than replayed history (include_contents="none").
    """
    diag = state.get("diagnosis_report", {}) or {}
    if not isinstance(diag, dict):
        diag = {}
    brief = state.get("diagnosis_brief", {}) or {}
    if not isinstance(brief, dict):
        brief = {}
    err = state.get("error_analysis_report", {}) or {}
    if not isinstance(err, dict):
        err = {}
    probe = state.get("latest_probe_metrics", {}) or {}

    parts = [
        f"BEST_OVERKILL_RATE: {state.get('best_overkill_rate')}",
        f"BEST_MISS_RATE: {state.get('best_miss_rate')}",
        f"DIAGNOSIS_REPORT: {json.dumps(diag, default=str)}",
        f"DIAGNOSIS_BRIEF: {json.dumps(brief, default=str)}",
        f"ERROR_ANALYSIS_REPORT: {json.dumps(err, default=str)}",
        f"LATEST_PROBE_METRICS: {json.dumps(probe, default=str)}",
        "REFINEMENT_POPULATION (summary, no scripts): "
        f"{json.dumps(_population_summary(state.get('refinement_population', [])), default=str)}",
    ]
    return "\n".join(parts)


_load_tried_tool = FunctionTool(func=load_tried_approaches_fn)

# ---------------------------------------------------------------------------
# FunctionTool — save strategy candidates
# ---------------------------------------------------------------------------

def _normalise_strategy_fingerprint(target_component: str, mechanism_class: str) -> dict:
    return {
        "target_component": (target_component or "unknown").strip().lower(),
        "mechanism_class": (mechanism_class or "unknown").strip().lower(),
    }


def _fingerprint_key(fingerprint: dict) -> tuple:
    return (
        (fingerprint or {}).get("target_component", "unknown"),
        (fingerprint or {}).get("mechanism_class", "unknown"),
    )


# Stereo-only failed fingerprints (mg8). For mono input these are irrelevant — not
# failures — so they must not block otherwise-valid mono approaches.
_STEREO_ONLY_FAILED_FINGERPRINTS = {
    ("stereo_fusion", "global_feature_difference_only"),
    ("stereo_fusion", "two_independent_backbone_stereo"),
}


def _is_known_failed_strategy_fingerprint(fingerprint: dict, input_modality: str = "stereo") -> bool:
    key = _fingerprint_key(_normalise_strategy_fingerprint(
        fingerprint.get("target_component", ""),
        fingerprint.get("mechanism_class", ""),
    ))
    blocklist = KNOWN_FAILED_STRATEGY_FINGERPRINTS
    if input_modality == "mono":
        blocklist = blocklist - _STEREO_ONLY_FAILED_FINGERPRINTS
    return key in blocklist


def _is_priority_correct_failure(approach: dict) -> bool:
    """Return True only when the strategy harmed P0/P1 objectives or crashed.

    A strategy whose failure_reason is "overkill_regression" improved miss_rate
    or recall but raised overkill. That means the direction is right — it just
    needs FP control added. Do NOT ban it; the planner can build on it with an
    FP-penalty component. Only P0/P1 regressions, crashes, and zero-movement
    runs are true failures that justify banning the fingerprint.
    """
    if not isinstance(approach, dict):
        return False
    result = approach.get("result", {}) or {}
    if result.get("improved") is not False:
        return False
    failure_reason = (approach.get("failure_reason") or "").strip().lower()
    # Overkill rose while miss/recall improved — useful direction, not a ban.
    # Use substring match + lower() so LLM-written variants ("Overkill_regression",
    # "overkill_regressed", etc.) are handled robustly.
    if "overkill_regression" in failure_reason or failure_reason == "overkill_regressed":
        return False
    return True


def _failed_strategy_fingerprint_keys(tried_approaches: list) -> set:
    keys = set()
    for approach in tried_approaches or []:
        if not _is_priority_correct_failure(approach):
            continue
        fingerprint = approach.get("strategy_fingerprint") or {}
        if not fingerprint:
            continue
        keys.add(_fingerprint_key(_normalise_strategy_fingerprint(
            fingerprint.get("target_component", ""),
            fingerprint.get("mechanism_class", ""),
        )))
    return keys


def _failed_attempt_count_for_target(tried_approaches: list, target_component: str) -> int:
    target = (target_component or "unknown").strip().lower()
    count = 0
    for approach in tried_approaches or []:
        if not isinstance(approach, dict):
            continue
        approach_target = (approach.get("target_component", "") or "").strip().lower()
        fingerprint = approach.get("strategy_fingerprint") or {}
        fingerprint_target = (fingerprint.get("target_component", "") or "").strip().lower()
        if target not in {approach_target, fingerprint_target}:
            continue
        # Only count true P0/P1 failures toward rotation lock — overkill regressions
        # indicate the target component is still worth pursuing with FP control.
        if _is_priority_correct_failure(approach):
            count += 1
    return count


def _failure_confidence(state: dict) -> str:
    diagnosis = state.get("diagnosis_report") or {}
    if not isinstance(diagnosis, dict):
        diagnosis = {}
    failure = diagnosis.get("failure_classification") or {}
    if not isinstance(failure, dict):
        failure = {}
    confidence = failure.get("confidence")
    if confidence:
        return str(confidence).strip().lower()

    brief = state.get("diagnosis_brief") or {}
    if not isinstance(brief, dict):
        return ""
    failure = brief.get("failure_classification") or {}
    if not isinstance(failure, dict):
        return ""
    return str(failure.get("confidence", "")).strip().lower()


def _has_preflight_probe_evidence(state: dict) -> bool:
    for key in ("latest_probe_metrics", "preflight_probe_results"):
        value = state.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
    return False


def _is_preflight_probe_fingerprint(fingerprint: dict) -> bool:
    target = (fingerprint or {}).get("target_component", "")
    mechanism = (fingerprint or {}).get("mechanism_class", "")
    values = {str(target).strip().lower(), str(mechanism).strip().lower()}
    return bool(values & {
        "preflight_probe",
        "diagnostic_probe",
        "feature_separability_probe",
        "loss_sensitivity_probe",
        "threshold_frontier_probe",
    })


def _preflight_probe_strategy() -> str:
    return (
        "preflight_probe: run cheap diagnostic probes before full refinement. "
        "Measure representation separability, loss sensitivity, and threshold-frontier "
        "viability, then print PROBE_METRICS with ng_recall, overkill_rate, "
        "G_prob_mean, NG_prob_mean, probability_gap, recommended_target_component, "
        "and should_continue."
    )


def save_strategy_candidates_fn(
    tool_context,
    strategy_a: str,
    strategy_b: str,
    strategy_c: str,
    selected: str,
    selection_reason: str,
    strategy_a_target_component: str = "",
    strategy_a_mechanism_class: str = "",
    strategy_b_target_component: str = "",
    strategy_b_mechanism_class: str = "",
    strategy_c_target_component: str = "",
    strategy_c_mechanism_class: str = "",
) -> str:
    """
    Save 3 mechanistically distinct refinement strategies and record which one
    to attempt in this inner iteration.

    Args:
        strategy_a: Name and one-sentence description of strategy A.
            Format: "name: description". e.g. "focal_loss: Replace BCE with
            focal loss alpha=0.75 gamma=2 to down-weight easy negatives."
        strategy_b: Name and one-sentence description of strategy B.
        strategy_c: Name and one-sentence description of strategy C.
        selected: Which strategy to implement this iteration — "a", "b", or "c".
        selection_reason: One sentence explaining why this strategy was selected
            (i.e. what evidence supports it and why the others were skipped).
        strategy_*_target_component: Structured target component for dedupe.
        strategy_*_mechanism_class: Structured mechanism class for dedupe.
    """
    n = int(tool_context.state.get("outer_iteration", 0))
    m = int(tool_context.state.get("inner_iteration", 0))

    candidates = {"a": strategy_a, "b": strategy_b, "c": strategy_c}
    fingerprints = {
        "a": _normalise_strategy_fingerprint(strategy_a_target_component, strategy_a_mechanism_class),
        "b": _normalise_strategy_fingerprint(strategy_b_target_component, strategy_b_mechanism_class),
        "c": _normalise_strategy_fingerprint(strategy_c_target_component, strategy_c_mechanism_class),
    }
    chosen_key = (selected or "a").strip().lower()
    if chosen_key not in candidates:
        chosen_key = "a"
    chosen_strategy = candidates[chosen_key]
    chosen_fingerprint = fingerprints[chosen_key]
    tried_approaches = tool_context.state.get("tried_approaches", [])

    if (
        _failure_confidence(tool_context.state) == "low"
        and not _has_preflight_probe_evidence(tool_context.state)
        and not _is_preflight_probe_fingerprint(chosen_fingerprint)
    ):
        return (
            "PREFLIGHT_PROBE_REQUIRED: diagnosis confidence is low and no "
            "pre-flight probe evidence is available. Select a preflight_probe "
            "strategy that runs cheap representation, loss-sensitivity, and "
            "threshold-frontier probes before any full refinement."
        )

    failed_count = _failed_attempt_count_for_target(
        tried_approaches,
        chosen_fingerprint["target_component"],
    )
    if (
        failed_count >= config.TARGET_COMPONENT_ROTATION_LOCK_FAILED_ATTEMPTS
        and not tool_context.state.get("target_rotation_override", False)
    ):
        return (
            "ROTATION_LOCK_REJECTED: selected target_component "
            f"'{chosen_fingerprint['target_component']}' has {failed_count} failed "
            "attempts without improvement. Choose a different target_component, or set "
            "state['target_rotation_override']=True only when threshold-curve and "
            "error-analysis evidence uniquely justify staying on this target."
        )

    failed_keys = _failed_strategy_fingerprint_keys(tried_approaches)
    input_modality = tool_context.state.get("input_modality", "stereo")
    if _is_known_failed_strategy_fingerprint(chosen_fingerprint, input_modality):
        return (
            "KNOWN_FAILED_STRATEGY_REJECTED: selected strategy fingerprint "
            f"{chosen_fingerprint} is in the small-data failed list from mini-goals "
            "7/8. Avoid threshold-only fixes, mg8 global feature-difference-only, "
            "larger backbones, and two-independent-backbone stereo."
        )

    if _fingerprint_key(chosen_fingerprint) in failed_keys:
        return (
            "DUPLICATE_STRATEGY_REJECTED: selected strategy fingerprint "
            f"{chosen_fingerprint} already failed. Choose a candidate with a new "
            "(target_component, mechanism_class) pair before coding."
        )

    tool_context.state["refinement_strategy_candidates"] = candidates
    tool_context.state["refinement_strategy_fingerprints"] = fingerprints
    tool_context.state["selected_refinement_strategy"] = chosen_strategy
    tool_context.state["selected_strategy_fingerprint"] = chosen_fingerprint
    tool_context.state["selected_refinement_strategy_key"] = chosen_key
    tool_context.state["selected_refinement_strategy_outer"] = n
    tool_context.state["selected_refinement_strategy_inner"] = m
    tool_context.state["strategy_selection_reason"] = selection_reason

    logger.info(
        "Planner (outer=%d, inner=%d): selected strategy '%s' — %s",
        n, m, chosen_key, chosen_strategy,
    )
    return (
        f"Strategy candidates saved (outer={n}, inner={m}). "
        f"Selected: '{chosen_key}' — {chosen_strategy}. "
        f"Fingerprint: {chosen_fingerprint}. "
        f"Reason: {selection_reason}"
    )


_save_candidates_tool = FunctionTool(func=save_strategy_candidates_fn)

# ---------------------------------------------------------------------------
# FunctionTool — guarantee a selected strategy before coding
# ---------------------------------------------------------------------------

def ensure_selected_strategy_fn(tool_context) -> str:
    """
    Ensure state["selected_refinement_strategy"] exists before the coder runs.
    The planner should set this, but the coder must not proceed with an empty
    strategy if the LLM forgot to call save_strategy_candidates_fn.
    """
    n = int(tool_context.state.get("outer_iteration", 0))
    m = int(tool_context.state.get("inner_iteration", 0))
    selected = (tool_context.state.get("selected_refinement_strategy") or "").strip()
    selected_outer = tool_context.state.get("selected_refinement_strategy_outer")
    selected_inner = tool_context.state.get("selected_refinement_strategy_inner")
    if selected and selected_outer == n and selected_inner == m:
        return f"STRATEGY_READY: {selected}"

    if selected:
        logger.warning(
            "Discarding stale strategy selected for outer=%s inner=%s while current is outer=%d inner=%d: %s",
            selected_outer,
            selected_inner,
            n,
            m,
            selected,
        )
        for key in (
            "refinement_strategy_candidates",
            "selected_refinement_strategy",
            "selected_refinement_strategy_key",
            "selected_strategy_fingerprint",
            "refinement_strategy_fingerprints",
            "selected_refinement_strategy_outer",
            "selected_refinement_strategy_inner",
            "strategy_selection_reason",
        ):
            tool_context.state.pop(key, None)

    diagnosis = tool_context.state.get("diagnosis_report") or {}
    if not isinstance(diagnosis, dict):
        diagnosis = {}
    error_report = tool_context.state.get("error_analysis_report") or {}
    if not isinstance(error_report, dict):
        error_report = {}
    instrumentation_required = bool(
        tool_context.state.get("error_analysis_instrumentation_required", False)
    )

    if instrumentation_required:
        fallback = (
            "instrumentation_repair: add mandatory PREDICTIONS, ERROR_ANALYSIS, "
            "EPOCH_LOG, CALIBRATION_STATS, and THRESHOLD_CURVE output without "
            "changing the ML strategy."
        )
        reason = "The previous run did not emit usable per-sample evidence."
        target = "instrumentation"
        mechanism = "fallback"
    elif (
        _failure_confidence(tool_context.state) == "low"
        and not _has_preflight_probe_evidence(tool_context.state)
    ):
        fallback = _preflight_probe_strategy()
        reason = "Diagnosis confidence is low and no pre-flight probe evidence exists."
        target = "preflight_probe"
        mechanism = "preflight_probe"
    else:
        target = (
            error_report.get("recommended_target_component")
            or diagnosis.get("target_component")
            or diagnosis.get("recommended_target")
            or "diagnosis_target"
        )
        recommended = (
            error_report.get("recommended_changes")
            or diagnosis.get("recommended_changes")
            or "implement the highest-priority diagnosis changes conservatively"
        )
        fallback = f"{target}_fallback: {recommended}"
        reason = "Planner did not persist a selected strategy; using diagnosis fallback."
        mechanism = "fallback"

    tool_context.state["refinement_strategy_candidates"] = {"fallback": fallback}
    tool_context.state["refinement_strategy_fingerprints"] = {
        "fallback": _normalise_strategy_fingerprint(target, mechanism)
    }
    tool_context.state["selected_refinement_strategy"] = fallback
    tool_context.state["selected_strategy_fingerprint"] = tool_context.state["refinement_strategy_fingerprints"]["fallback"]
    tool_context.state["selected_refinement_strategy_key"] = "fallback"
    tool_context.state["selected_refinement_strategy_outer"] = n
    tool_context.state["selected_refinement_strategy_inner"] = m
    tool_context.state["strategy_selection_reason"] = reason
    logger.warning("Strategy gate supplied fallback strategy: %s", fallback)
    return f"STRATEGY_FALLBACK: {fallback} Reason: {reason}"


_ensure_strategy_tool = FunctionTool(func=ensure_selected_strategy_fn)

# ---------------------------------------------------------------------------
# Planner instruction
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are the Refinement Planner Agent for Phase 2.

Your role is to propose 3 mechanistically distinct strategies for the current
target component and select which one to attempt this inner iteration, based
on what has already been tried and failed.

---
## STEP 1 — Restore history and context

Call `load_tried_approaches_fn` immediately. This ensures tried_approaches is
populated even after a session restart or retry reset. Do not skip this step.
Its return value ALSO contains the context you need for STEP 2 — read it directly
from the tool output rather than from conversation history:
- DIAGNOSIS_REPORT  — target_component, recommended_changes, failure_classification,
  prediction
- DIAGNOSIS_BRIEF   — failure_classification, threshold_curve, threshold_curve_summary,
  and deterministic recommendations
- ERROR_ANALYSIS_REPORT — dominant_failure, recommended_target_component,
  recommended_changes (may be empty on first inner iteration)
- LATEST_PROBE_METRICS — pre-flight probe evidence from the last probe-only run,
  including probability_gap and recommended_target_component

---
## STEP 2 — Read context

From the `load_tried_approaches_fn` output (STEP 1), use:
- DIAGNOSIS_REPORT, DIAGNOSIS_BRIEF, ERROR_ANALYSIS_REPORT, LATEST_PROBE_METRICS
  as described above.
- REFINEMENT_POPULATION (summary) — small archive of accepted improvements and
  lower-overkill non-best candidates (metrics only). Use it to avoid single-chain
  tunnel vision: if one archived candidate has much lower overkill but slightly
  worse recall, propose a strategy that preserves its FP control while recovering recall.
- TRIED_APPROACHES_RECENT — the last few prior attempts in full (from the load
    tool output). Each entry has: outer, inner, target_component, changes_summary,
    strategy_fingerprint, result.ng_recall, result.overkill,
    result.improved, failure_reason, prediction_verification
- TRIED_APPROACHES_AGGREGATE — summary over ALL prior attempts:
    total_attempts, failed_fingerprints (list of {target_component, mechanism_class}),
    failure_counts_by_reason, failed_attempts_by_target. Use this for older history
    beyond the recent window when checking for repeats and rotation locks.

---
## STEP 3 — Identify what has already been tried and failed

Scan TRIED_APPROACHES_RECENT and TRIED_APPROACHES_AGGREGATE carefully. For each
failed attempt (`result.improved = False`, and in the aggregate's
`failed_fingerprints`), extract both `target_component` and `mechanism_class`. The
same `(target_component, mechanism_class)` pair MUST NOT be selected again. Also
read `changes_summary` in the recent entries to avoid prose-level repeats.

If `failed_attempts_by_target` shows 6 or more failed attempts for the same target
component, treat that target as rotation-locked. Select a different target unless threshold-curve and
error-analysis evidence uniquely justify staying. If you must stay on a
rotation-locked target, set state["target_rotation_override"] = True before
calling `save_strategy_candidates_fn` and explain the evidence in
`selection_reason`.

Identify the most common failure reasons:
- "overkill_regression" — miss_rate/recall IMPROVED but overkill rose. This is
  a PARTIAL result, not a true failure. The strategy direction is correct — it
  is NOT banned. You may retry this fingerprint with explicit FP control added
  (e.g. combine it with fp_penalty_loss or a constrained threshold stage).
- "recall_regression" — the change worsened NG recall / raised miss_rate. True
  P0 failure — ban this fingerprint and do not retry this target if it has
  accumulated 6 such failures (rotation lock).
- "execution_failed" — script crashed (LLM code generation issue). Banned.
- "no_improvement" — metrics did not move at all. Banned.
- "diagnosis_prediction_failed" — the attempted mechanism violated the
  diagnosis prediction; do not select the same mechanism again unless a new
  probe gives stronger evidence.

If diagnosis confidence is "low" and `state["latest_probe_metrics"]` is absent,
you MUST select a `preflight_probe` strategy before any normal full refinement.
The deterministic gate rejects normal strategies in this state.

Known failed strategy fingerprints are banned before candidate selection:
- (`threshold_sweep`, `threshold_acceptance_distance`) — mg7 threshold tuning plateaued.
- (`calibration`, `threshold_only`) — Calibration/threshold curves are REPORTING, not the only fix.
- (`stereo_fusion`, `global_feature_difference_only`) — mg8 global feature-difference-only hurt.
- (`stereo_fusion`, `two_independent_backbone_stereo`) — two-independent-backbone stereo adds capacity and overfits.
- (`model_architecture`, `larger_backbone`) — larger backbone is the wrong lever for ~287 train samples.
- full-freeze-only is not banned globally, but if prior evidence shows flat
  predictions or `prob_gap` collapse around 0.02, treat it as underfit and select
  partial-unfreeze of the final ResNet block/layer4 instead.

---
## STEP 4 — Propose 3 diverse strategies

Determine the active target component first:
- If `error_analysis_report.recommended_target_component` is present, it is the
  active target. It is based on per-sample FP/FN evidence and overrides
  `diagnosis_report.target_component`.
- Else if `latest_probe_metrics.recommended_target_component` is present, it is
  the active target. It is based on cheap probe evidence and overrides low-confidence
  diagnosis prose.
- Otherwise use `diagnosis_report.target_component`.

For the active target component, propose exactly 3 strategies that are
mechanistically distinct from each other, from the recent `changes_summary` entries
in TRIED_APPROACHES_RECENT, and from every `(target_component, mechanism_class)` pair
in TRIED_APPROACHES_AGGREGATE.failed_fingerprints.

small-data-safe strategy priority for this project:
1. freeze or partially-freeze the pretrained backbone + small head, with
   partial-unfreeze adaptation preferred when full freeze underfits: freeze early backbone
   layers, unfreeze only the final ResNet block/layer4 or equivalent, and use a
   small regularized head;
2. weight decay + dropout;
3. AOI-safe augmentation;
4. smaller/regularized head;
5. local/patch or localized L/R difference evidence.

De-prioritize: larger backbone; two-independent-backbone stereo; global
feature-difference-only; any mg7/mg8 failed fingerprint listed above. AOI-safe
augmentation means geometric transforms are sampled once and applied IDENTICALLY
to L and R. Exclude heavy color jitter, random erasing over defects, random
perspective, and any L/R-desynchronizing affine/rotation/crop.

If DIAGNOSIS_BRIEF.failure_classification.failure_mode is `full_freeze_underfit`,
or latest probe/calibration evidence shows `prob_gap` near zero, prefer a
`partial_unfreeze_last_block` mechanism over full-freeze-only. Keep ResNet18
shared weights, unfreeze only layer4/final block, use a small dropout head,
AdamW, and nonzero weight_decay. Prefer local/patch/ROI evidence over another
global feature-difference-only attempt.

When `diagnosis_brief.threshold_curve` is present, use it directly to decide
whether the current failure is separability, calibration, or threshold policy:
- If no threshold point gets close to the acceptance overkill/accuracy target
  while preserving miss_rate <= 0.03, prefer feature/loss/data strategies.
- If the curve contains a near-accepted operating point, prefer calibration or
  threshold-selection strategies.

When REFINEMENT_POPULATION (summary) contains lower-overkill candidates, treat them as
useful population members even if they were not accepted as the global best.
Prefer strategies that combine their overkill control with the current best
pipeline's recall, rather than restarting from only the single best script.

Each strategy must differ in its fundamental mechanism — not just in a
hyperparameter. Examples of mechanistically distinct strategies:

For target "weighted_loss":
  A. Focal loss (change loss function kernel)
  B. Class-balanced oversampling in DataLoader (change data distribution)
  C. Dynamic FP penalty: fp_weight = 1 + 5 * max(0, overkill - 0.08) (constraint-aware loss)

For target "stereo_fusion":
  A. Absolute difference map abs(L-R) as extra channel (explicit diff features)
  B. Cross-attention between L and R feature maps (learned alignment)
  C. Spatial pyramid of L-R differences at 3 scales (multi-scale diff)

For target "model_architecture":
  A. Swap backbone to ResNet34 (more capacity)
  B. Add channel-attention (SE block) after fusion layer (feature reweighting)
  C. Add auxiliary classification head on stereo diff branch (multi-task regularization)

For target "calibration":
  A. Temperature scaling on logits post-training (platt scaling)
  B. Label smoothing during training (soft targets reduce overconfidence)
  C. Isotonic regression calibration on val set probabilities (non-parametric)

For target "preprocessing_normalization":
  A. Per-lot mean/std normalization computed from train set only
  B. CLAHE contrast equalization applied before stereo fusion
  C. Histogram matching: align each image to a reference lot's distribution

For target "threshold_sweep":
  A. Finer sweep step 0.01 with Youden-J statistic for selection
  B. Two-stage constrained sweep: first find zero-FN region, then minimize FP
  C. Per-lot threshold: sweep separately on each lot's val samples

For low-confidence diagnosis before probes:
  A. preflight_probe: run representation separability, loss sensitivity, and
     threshold-frontier probes only; print PROBE_METRICS with recommended target.
  B. feature_separability_probe: frozen-feature linear/MLP probe.
  C. loss_sensitivity_probe: 1-epoch loss-direction comparison.

If the previous attempt regressed recall or miss_rate, do not select another
strategy from the same target component unless all error-analysis options are
exhausted. A strategy that reduces overkill by increasing miss_rate is a failed
trade-off, not progress.

---
## STEP 5 — Select which strategy to try next

Choose the strategy that:
1. Has NOT been tried before (check tried_approaches)
2. Directly addresses the dominant failure in error_analysis_report (if available)
3. Is the most likely to reduce the specific failure mode:
   - overkill_regression → pick a strategy that constrains FP
   - recall_regression → pick a strategy that preserves TP
   - execution_failed → pick a simpler, more standard implementation
   - no_improvement → pick the most architecturally different option

If all 3 strategies were tried before, still select the one with the best
prior result (smallest acceptance_distance, even if not improved).

---
## STEP 6 — Call save_strategy_candidates_fn

Call `save_strategy_candidates_fn` with all arguments:
- `strategy_a`, `strategy_b`, `strategy_c` — concise name: description strings
- `selected` — "a", "b", or "c"
- `selection_reason` — one sentence: what evidence points to this strategy
- `strategy_a_target_component`, `strategy_a_mechanism_class`
- `strategy_b_target_component`, `strategy_b_mechanism_class`
- `strategy_c_target_component`, `strategy_c_mechanism_class`

Use stable lowercase enum-like mechanism classes, for example:
`temperature_scaling`, `isotonic_calibration`, `class_balanced_sampler`,
`focal_loss`, `fp_penalty_loss`, `stereo_diff_features`, `lot_normalization`,
`threshold_acceptance_distance`, `training_schedule`.

If the tool returns `DUPLICATE_STRATEGY_REJECTED`, select a different candidate
with a new fingerprint and call the tool again before moving on.

If the tool returns `ROTATION_LOCK_REJECTED`, select a strategy from a different
target component unless there is uniquely strong evidence to override the lock.

---
## STEP 7 — Summarise

After the tool call, output a 2-line summary:
1. The 3 candidates (one line each)
2. Which one was selected and why
"""

# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------

refinement_planner_agent = LlmAgent(
    name="refinement_planner_agent",
    model=config.MODEL_PRO,
    description=(
        "Reads tried_approaches, diagnosis_report, and error_analysis_report to "
        "propose 3 mechanistically distinct refinement strategies and select which "
        "one to attempt next. Writes state['selected_refinement_strategy']."
    ),
    instruction=_INSTRUCTION,
    tools=[_load_tried_tool, _save_candidates_tool],
    include_contents="none",
    before_model_callback=[log_context_size_callback, budget_stop_callback],
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)


_GATE_INSTRUCTION = """You are the Strategy Gate Agent.

Your only job is to ensure `state["selected_refinement_strategy"]` exists before
the refinement coder runs.

Call `ensure_selected_strategy_fn` immediately. Do not call any other tools.
After the tool returns, output exactly the returned status string.
"""


strategy_gate_agent = LlmAgent(
    name="strategy_gate_agent",
    model=config.MODEL,
    description=(
        "Deterministically verifies that the planner selected a refinement strategy. "
        "Writes a fallback strategy if the planner did not persist one."
    ),
    instruction=_GATE_INSTRUCTION,
    tools=[_ensure_strategy_tool],
    include_contents="none",
    before_model_callback=log_context_size_callback,
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
