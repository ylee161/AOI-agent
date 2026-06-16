import hashlib
import json
import logging
import tempfile
from pathlib import Path

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
)
from mle_star_agent.shared import code_runner, loop_guard, metric_guard
from mle_star_agent.shared.code_diff import make_code_diff
from mle_star_agent.shared.callbacks import (
    count_tokens_callback,
    log_context_size_callback,
    rate_limit_retry_callback,
)
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.regression_guard import regression_blocked
from mle_star_agent.shared.selection_metrics import (
    AVERAGED_EVALUATION_KEY,
    average_metrics_dicts,
    build_selection_evaluation,
)
from mle_star_agent.shared.aoi_smoke_triage import build_smoke_diagnostics
from mle_star_agent.shared.data_split import board_grouped_kfold, stratified_kfold
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
from mle_star_agent.shared.curve_extrapolation import project_power_law
from mle_star_agent.phases.phase2_refinement.ideator_agent import trigger_ideation
from mle_star_agent.phases.phase2_refinement import fusion, warm_restart

logger = logging.getLogger(__name__)

OPTIMIZER_LR_SCHEDULE_TARGET = "optimizer/lr-schedule"
_TARGET_COMPONENT_ALIASES = {
    "optimizer_lr_schedule": OPTIMIZER_LR_SCHEDULE_TARGET,
    "optimizer lr schedule": OPTIMIZER_LR_SCHEDULE_TARGET,
    "lr_schedule": OPTIMIZER_LR_SCHEDULE_TARGET,
    "lr-schedule": OPTIMIZER_LR_SCHEDULE_TARGET,
    "training_schedule": OPTIMIZER_LR_SCHEDULE_TARGET,
}

# Persistent AOI KB (cross-run "Experience-Driven Global Memory") bounds.
# The KB is append-ordered; without a cap it grows unbounded across runs and the
# planner's TAG_ROLLUP gets dominated by stale dead-ends. Keep the most recent
# PERSISTENT_KB_MAX_RECORDS via FIFO, and skip appending a record whose
# (tags, target_component, mechanism_class) signature matches any of the last
# PERSISTENT_KB_DEDUP_RECENT_WINDOW records so repeated identical attempts don't
# flood the memory.
PERSISTENT_KB_MAX_RECORDS = 200
PERSISTENT_KB_DEDUP_RECENT_WINDOW = 25


def _kb_dedup_signature(record: dict) -> tuple:
    """Order-insensitive (tags, target_component, mechanism_class) signature.

    Legacy records lack the fingerprint fields, so they resolve to (tags, None,
    None) and only collide with another fingerprint-less record — never with a
    new fingerprinted one. Never raises on malformed input.
    """
    if not isinstance(record, dict):
        return ((), None, None)
    tags = record.get("tags") or []
    tags_key = tuple(sorted(str(t) for t in tags if t)) if isinstance(tags, list) else ()
    return (tags_key, record.get("target_component"), record.get("mechanism_class"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_pop(state, key, default=None):
    """ADK State doesn't support .pop() or del — null the key out instead."""
    val = state.get(key, default)
    if key in state:
        state[key] = None
    return val


def _normalise_target_component(target_component: str) -> str:
    target = (target_component or "unknown").strip().lower()
    return _TARGET_COMPONENT_ALIASES.get(target, target)


def _attempt_label(target_component: str, fingerprint: dict | None) -> str:
    target = _normalise_target_component(target_component)
    mechanism = ((fingerprint or {}).get("mechanism_class") or "unknown").strip().lower()
    return f"{target}:{mechanism}"


def _dedupe_nonempty_strings(values) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _mechanism_label(target_component: str | None, mechanism_class: str | None) -> str:
    target = _normalise_target_component(str(target_component or "unknown"))
    mechanism = str(mechanism_class or "").strip().lower()
    if mechanism and mechanism != "unknown":
        return f"{target}:{mechanism}"
    return target if target != "unknown" else ""


def _recommended_target_from_state(state: dict) -> str:
    """Read diagnosis_brief.recommended_target without assuming schema quality."""
    if not hasattr(state, "get"):
        return ""
    brief = state.get("diagnosis_brief") or {}
    if isinstance(brief, dict):
        target = str(brief.get("recommended_target") or "").strip()
        if target:
            return _normalise_target_component(target)
    return ""


def _helpful_mechanisms_from_state(
    state: dict,
    *,
    improved: bool,
    selected_fingerprint: dict | None,
    code_diff: str,
) -> list[str]:
    """Mechanisms attached to an accepted diff; empty on missing evidence."""
    if not improved or not code_diff:
        return []
    fingerprint = selected_fingerprint if isinstance(selected_fingerprint, dict) else {}
    target = _recommended_target_from_state(state)
    if not target:
        return []
    return _dedupe_nonempty_strings([
        _mechanism_label(target, fingerprint.get("mechanism_class")),
    ])


def _harmful_mechanisms_from_tried_approaches(
    tried_approaches: list,
    outer_iteration: int,
) -> list[str]:
    """Failed mechanisms in the current outer cycle since the last accepted entry."""
    entries = [e for e in (tried_approaches or []) if isinstance(e, dict)]
    current_outer = []
    for entry in entries:
        try:
            if int(entry.get("outer", -1)) == int(outer_iteration):
                current_outer.append(entry)
        except (TypeError, ValueError):
            continue

    last_accepted_idx = -1
    for idx, entry in enumerate(current_outer):
        result = entry.get("result") or {}
        if result.get("improved") is True or entry.get("failure_reason") == "accepted":
            last_accepted_idx = idx

    rejected = []
    for entry in current_outer[last_accepted_idx + 1:]:
        result = entry.get("result") or {}
        failure_reason = str(entry.get("failure_reason") or "").strip().lower()
        rejected_reason = failure_reason in {
            "no_improvement",
            "recall_regression",
            "overkill_regression",
        }
        if result.get("improved") is not False and not rejected_reason:
            continue
        fingerprint = entry.get("strategy_fingerprint") or {}
        target = (
            entry.get("target_component")
            or (fingerprint if isinstance(fingerprint, dict) else {}).get("target_component")
        )
        mechanism = (
            fingerprint if isinstance(fingerprint, dict) else {}
        ).get("mechanism_class")
        rejected.append(_mechanism_label(target, mechanism))
    return _dedupe_nonempty_strings(rejected)

def _save_best_pipeline(state: dict) -> None:
    """Persist the authoritative resume snapshot to best_pipeline.json.

    Anti-regression guard: if the persistent KB declares a hard floor
    (an ``is_floor`` entry with ``floor_score.roc_auc``), refuse to overwrite a
    proven best_pipeline.json with a candidate scoring below that floor. This
    prevents a cold/collapsed run from clobbering an all-time-best result
    (the 0.697 -> 0.067 regression this guard was added to fix).
    """
    current_best_score = float(state.get("current_best_score", 0.0) or 0.0)
    best_miss_rate = state.get("best_miss_rate")
    if best_miss_rate is None:
        best_miss_rate = max(0.0, 1.0 - current_best_score)

    candidate_roc_auc = float(state.get("best_roc_auc", 0.0) or 0.0)
    blocked, floor = regression_blocked(candidate_roc_auc)
    if blocked:
        logger.warning(
            "REGRESSION BLOCKED: skipping best_pipeline.json write "
            "(candidate roc_auc %.4f < KB floor %.4f)",
            candidate_roc_auc,
            floor,
        )
        return

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
        "best_selection_evaluation": state.get("best_selection_evaluation"),
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


def _clear_ablation_state(state) -> None:
    keys = list(state.to_dict().keys()) if hasattr(state, "to_dict") else list(state.keys())
    for key in keys:
        if key == "ablation_results" or key.startswith("ablation_result_") or key.startswith("ablation_script_"):
            _state_pop(state, key)


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
    return _metrics_from_dict(average_metrics_dicts(successful)), seed_results


def _aggregate_cv_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold validation metrics into conservative decision metrics."""
    if not fold_metrics:
        return {}

    def values(key: str) -> list[float]:
        return [float(metrics.get(key, 0.0) or 0.0) for metrics in fold_metrics]

    ng_recall = values("ng_recall")
    overkill = values("overkill_rate")
    accuracy = values("accuracy")
    miss_rate = values("miss_rate")
    f1 = values("f1")

    aggregated = {
        "cv_fold_count": len(fold_metrics),
        "mean_val_ng_recall": sum(ng_recall) / len(ng_recall),
        "worst_fold_val_ng_recall": min(ng_recall),
        "mean_val_overkill": sum(overkill) / len(overkill),
        "worst_fold_val_overkill": max(overkill),
        "mean_val_accuracy": sum(accuracy) / len(accuracy),
        "worst_fold_val_accuracy": min(accuracy),
        "mean_val_miss_rate": sum(miss_rate) / len(miss_rate),
        "worst_fold_val_miss_rate": max(miss_rate),
    }
    if f1:
        aggregated["mean_val_f1"] = sum(f1) / len(f1)
        aggregated["worst_fold_val_f1"] = min(f1)

    # Compatibility aliases consumed by existing AOIMetrics/state code. The
    # acceptance layer maps CV dicts the same way for direct dict comparisons.
    aggregated.update({
        "ng_recall": aggregated["worst_fold_val_ng_recall"],
        "overkill_rate": aggregated["mean_val_overkill"],
        "accuracy": aggregated["mean_val_accuracy"],
        "miss_rate": aggregated["worst_fold_val_miss_rate"],
        "f1": aggregated.get("mean_val_f1", 0.0),
    })
    return aggregated


def _script_with_data_split_path(script: str, split_path: Path) -> str:
    path_text = str(split_path)
    patched = script
    for old in {
        str(config.CKPT_DATA_SPLIT),
        "checkpoints/data_split_grouped.json",
        "checkpoints/data_split.json",
    }:
        patched = patched.replace(old, path_text)
    return patched


def _fold_split_payload(base_split: dict, train_df, val_df, fold_index: int, test_rows: list[dict] | None = None, cv_mode: str = "board_grouped_kfold") -> dict:
    train_rows = train_df.to_dict("records")
    val_rows = val_df.to_dict("records")
    test_rows = list(test_rows or [])
    all_rows = train_rows + val_rows + test_rows
    labels = [row.get("label") for row in all_rows]
    stats = dict(base_split.get("stats", {}) or {})
    stats.update({
        "cv_fold": fold_index,
        "total": len(all_rows),
        "ng_count": labels.count("NG"),
        "g_count": labels.count("G"),
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "board_groups": sorted({row.get("board_code") for row in all_rows if row.get("board_code")}),
        "val_board_groups": sorted({row.get("board_code") for row in val_rows if row.get("board_code")}),
    })
    metadata = dict(base_split.get("metadata", {}) or {})
    metadata.update({"cv_fold": fold_index, "cv_mode": cv_mode})
    return {
        "metadata": metadata,
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
        "stats": stats,
    }


def _run_board_grouped_cv_confirmation(script: str, k: int = 3) -> tuple[dict | None, list[dict]]:
    """Run one promising candidate over cross-validation folds.

    The fold scheme matches the active split strategy so the confirmation
    measures the SAME regime the candidate was trained/selected under:
      * grouped → board-grouped k-fold (cross-lot holdout per fold)
      * mixed   → label-stratified k-fold (lots mixed across train/val)
    Using grouped folds to confirm a mixed candidate would re-impose the
    cross-lot wall and can wrongly reject an in-distribution candidate.
    """
    base_split = load_checkpoint(config.CKPT_DATA_SPLIT)
    strategy = (
        base_split.get("metadata", {}).get("split_strategy")
        or getattr(config, "SPLIT_STRATEGY", "grouped")
    )
    cv_samples = (
        list(base_split.get("train", []) or [])
        + list(base_split.get("val", []) or [])
        + list(base_split.get("test", []) or [])
    )
    if strategy == "mixed":
        folds = stratified_kfold(cv_samples, k=k)
        cv_mode = "stratified_kfold"
    else:
        folds = board_grouped_kfold(cv_samples, k=k)
        cv_mode = "board_grouped_kfold"
    fold_results = []

    for fold_index, (train_df, val_df) in enumerate(folds, start=1):
        fold_payload = _fold_split_payload(base_split, train_df, val_df, fold_index, test_rows=[], cv_mode=cv_mode)
        with tempfile.NamedTemporaryFile(
            suffix=f"_aoi_cv_fold_{fold_index}.json",
            mode="w",
            delete=False,
        ) as handle:
            json.dump(fold_payload, handle)
            fold_path = Path(handle.name)

        try:
            result = code_runner.run_script(
                _script_with_data_split_path(script, fold_path),
                timeout=config.TIMEOUT_SECONDS,
                env={"AOI_RANDOM_SEED": "42", "PYTHONHASHSEED": "42", "SEED": "42"},
            )
        finally:
            fold_path.unlink(missing_ok=True)

        metrics = parse_metrics(result.stdout)
        metrics = metric_guard.guard_metrics(
            metrics, result.duration_ms, context=f"phase2 board-cv fold={fold_index}"
        )
        metrics_dict = metrics_to_dict(metrics) if metrics else None
        fold_results.append({
            "fold": fold_index,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration_ms": round(result.duration_ms, 1),
            "val_board_groups": fold_payload["stats"]["val_board_groups"],
            "train_size": fold_payload["stats"]["train_size"],
            "val_size": fold_payload["stats"]["val_size"],
            "metrics": metrics_dict,
            "stderr_tail": result.stderr[-1000:],
        })
        if metrics_dict:
            logger.info(
                "Board CV fold %d/%d: ng_recall=%.4f miss_rate=%.4f "
                "overkill=%.4f accuracy=%.4f val_boards=%s",
                fold_index, k,
                float(metrics_dict.get("ng_recall", 0.0) or 0.0),
                float(metrics_dict.get("miss_rate", 1.0) or 1.0),
                float(metrics_dict.get("overkill_rate", 1.0) or 1.0),
                float(metrics_dict.get("accuracy", 0.0) or 0.0),
                fold_payload["stats"]["val_board_groups"],
            )
        else:
            logger.warning(
                "Board CV fold %d/%d failed: rc=%s timed_out=%s stderr=%s",
                fold_index, k, result.returncode, result.timed_out,
                result.stderr[-300:],
            )

    successful = [r["metrics"] for r in fold_results if r.get("metrics") is not None]
    if len(successful) != k:
        return None, fold_results
    return _aggregate_cv_fold_metrics(successful), fold_results


def _format_cv_evaluation(selection_evaluation: dict | None) -> str:
    def metric_value(metrics: dict, key: str, default: float) -> float:
        value = metrics.get(key, default)
        if value is None:
            return default
        return float(value)

    if not isinstance(selection_evaluation, dict):
        return ""
    if selection_evaluation.get("mode") != "board_grouped_cv":
        return ""
    lines = ["BOARD_GROUPED_CV:"]
    for fold in selection_evaluation.get("per_fold", []) or []:
        metrics = fold.get("metrics") or {}
        if metrics:
            lines.append(
                "  fold {fold}: ng_recall={ng:.3f} miss_rate={miss:.3f} "
                "overkill={over:.3f} accuracy={acc:.3f} val_boards={boards}".format(
                    fold=fold.get("fold"),
                    ng=metric_value(metrics, "ng_recall", 0.0),
                    miss=metric_value(metrics, "miss_rate", 1.0),
                    over=metric_value(metrics, "overkill_rate", 1.0),
                    acc=metric_value(metrics, "accuracy", 0.0),
                    boards=fold.get("val_board_groups", []),
                )
            )
        else:
            lines.append(
                "  fold {fold}: FAILED rc={rc} timed_out={timed_out}".format(
                    fold=fold.get("fold"),
                    rc=fold.get("returncode"),
                    timed_out=fold.get("timed_out"),
                )
            )
    metrics = selection_evaluation.get("metrics") or {}
    if metrics:
        lines.append(
            "  aggregate: worst_ng_recall={ng:.3f} worst_miss_rate={miss:.3f} "
            "mean_overkill={over:.3f} mean_accuracy={acc:.3f}".format(
                ng=metric_value(metrics, "worst_fold_val_ng_recall", 0.0),
                miss=metric_value(metrics, "worst_fold_val_miss_rate", 1.0),
                over=metric_value(metrics, "mean_val_overkill", 1.0),
                acc=metric_value(metrics, "mean_val_accuracy", 0.0),
            )
        )
    return "\n".join(lines)


def _confirm_improvement_with_selection_average(
    *,
    script: str,
    metrics,
    current_metrics: dict,
    initially_improved: bool,
    run_average=None,
) -> tuple[bool, dict, dict | None]:
    metrics_dict = _normalise_metrics_dict(metrics)
    if not _requires_multiseed_confirmation(metrics_dict, initially_improved):
        return initially_improved, metrics_dict, None

    if run_average is not None:
        averaged_metrics, seed_results = run_average(script)
        selection_evaluation = build_selection_evaluation(
            seeds=config.MULTISEED_CONFIRMATION_SEEDS,
            seed_results=seed_results,
            averaged_metrics=_normalise_metrics_dict(averaged_metrics) if averaged_metrics is not None else None,
        )
    else:
        try:
            averaged_metrics, fold_results = _run_board_grouped_cv_confirmation(script, k=3)
            cv_failure_reason = "one_or_more_cv_folds_failed"
        except (RuntimeError, ValueError) as exc:
            averaged_metrics = None
            fold_results = []
            cv_failure_reason = str(exc)
        selection_evaluation = {
            "status": "success" if averaged_metrics is not None else "incomplete",
            "mode": "board_grouped_cv",
            "fold_count": 3,
            "metrics": dict(averaged_metrics) if averaged_metrics is not None else None,
            "per_fold": fold_results,
            "successful_fold_count": sum(1 for r in fold_results if r.get("metrics") is not None),
            "expected_fold_count": 3,
        }
        if averaged_metrics is None:
            selection_evaluation["failure_reason"] = cv_failure_reason
    if selection_evaluation.get("status") != "success":
        return False, metrics_dict, selection_evaluation

    selected_metrics = dict(selection_evaluation["metrics"])
    improved = is_acceptance_improvement(selected_metrics, current_metrics)
    return improved, selected_metrics, selection_evaluation


# ---------------------------------------------------------------------------
# Pareto archive maintenance for the refinement population
#
# The four refinement objectives are:
#   miss_rate     (lower better)
#   overkill_rate (lower better)
#   ng_recall     (higher better)
#   accuracy      (higher better)
#
# We keep a true Pareto archive instead of a scalar-sorted top-N: a candidate is
# only discarded when it is dominated by an existing member, and members are only
# dropped when the new candidate dominates them. When the archive overflows
# REFINEMENT_POPULATION_MAX we evict the member contributing the least
# hypervolume (the most redundant point on the front).
# ---------------------------------------------------------------------------

# Reference (nadir) point in the all-minimisation objective space:
#   (miss_rate, overkill_rate, -ng_recall, -accuracy).
# Every real metric vector is component-wise <= this worst-case point, so each
# member's hypervolume box is non-negative.
_PARETO_REFERENCE = (1.0, 1.0, 0.0, 0.0)


def _safe_float(value, default: float) -> float:
    """Coerce ``value`` to float, treating only ``None`` (not a legitimate 0.0)
    as missing. Using ``x or default`` here would corrupt a real best-case 0.0
    miss/overkill into the worst-case default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pareto_objective_vector(metrics: dict) -> tuple[float, float, float, float]:
    """Map metrics to an all-minimisation objective vector (lower is better)."""
    return (
        _safe_float(metrics.get("miss_rate"), 1.0),
        _safe_float(metrics.get("overkill_rate"), 1.0),
        -_safe_float(metrics.get("ng_recall"), 0.0),
        -_safe_float(metrics.get("accuracy"), 0.0),
    )


def _dominates(a: tuple, b: tuple) -> bool:
    """True if vector ``a`` Pareto-dominates ``b`` (all-minimisation): no worse in
    every objective and strictly better in at least one."""
    no_worse = all(ai <= bi for ai, bi in zip(a, b))
    strictly_better = any(ai < bi for ai, bi in zip(a, b))
    return no_worse and strictly_better


def _hypervolume(vectors: list[tuple]) -> float:
    """Exact hypervolume of the union of boxes [v, reference] via inclusion-exclusion.

    Each point ``v`` (all-minimisation) dominates the axis-aligned box bounded
    above by ``_PARETO_REFERENCE``. The dominated volume is the union of those
    boxes. Inclusion-exclusion over subsets is exact and cheap for the small
    archive sizes here (<= REFINEMENT_POPULATION_MAX + 1 points).
    """
    n = len(vectors)
    if n == 0:
        return 0.0
    total = 0.0
    for mask in range(1, 1 << n):
        # Intersection of the selected boxes: per axis the lower bound is the
        # worst (max) coordinate over the subset; volume is product of
        # (reference - lower_bound), clamped at 0.
        lower = [-float("inf")] * len(_PARETO_REFERENCE)
        bits = 0
        for i in range(n):
            if mask & (1 << i):
                bits += 1
                for axis, val in enumerate(vectors[i]):
                    if val > lower[axis]:
                        lower[axis] = val
        vol = 1.0
        for axis, ref in enumerate(_PARETO_REFERENCE):
            edge = ref - lower[axis]
            if edge <= 0.0:
                vol = 0.0
                break
            vol *= edge
        total += vol if (bits % 2 == 1) else -vol
    return total


def _least_hypervolume_contributor(population: list[dict]) -> int:
    """Index of the member whose removal shrinks the front's hypervolume least.

    Ties (e.g. fully degenerate vectors) fall back to the largest crowding
    distance — the most-isolated member is the safest to drop only when no
    member contributes measurable volume."""
    vectors = [_pareto_objective_vector(p.get("metrics", {})) for p in population]
    total_hv = _hypervolume(vectors)
    contributions = []
    for i in range(len(vectors)):
        without = vectors[:i] + vectors[i + 1:]
        contributions.append(total_hv - _hypervolume(without))
    min_contrib = min(contributions)
    candidates = [i for i, c in enumerate(contributions) if c <= min_contrib + 1e-12]
    if len(candidates) == 1:
        return candidates[0]
    # Fallback tiebreaker: crowding distance over the tied members. Evict the one
    # with the LARGEST crowding distance (most isolated -> least informative to a
    # diversity-seeking selector once volume is uninformative).
    crowd = _crowding_distances(vectors)
    return max(candidates, key=lambda i: crowd[i])


def _crowding_distances(vectors: list[tuple]) -> list[float]:
    """NSGA-II crowding distance per point (boundary points get +inf)."""
    n = len(vectors)
    distances = [0.0] * n
    if n <= 2:
        return [float("inf")] * n
    num_obj = len(vectors[0])
    for axis in range(num_obj):
        order = sorted(range(n), key=lambda i: vectors[i][axis])
        lo = vectors[order[0]][axis]
        hi = vectors[order[-1]][axis]
        span = hi - lo
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")
        if span <= 0:
            continue
        for k in range(1, n - 1):
            prev_v = vectors[order[k - 1]][axis]
            next_v = vectors[order[k + 1]][axis]
            distances[order[k]] += (next_v - prev_v) / span
    return distances


def _update_refinement_population(
    state: dict,
    script: str,
    metrics,
    current_metrics: dict,
    improved: bool,
    outer_iteration: int,
    inner_iteration: int,
) -> None:
    """Maintain a true Pareto archive of useful candidates so refinement is not a
    single chain.

    The candidate is first gated on eligibility (an accepted improvement, or a
    lower-overkill divergent branch). It is then merged into the archive by
    Pareto non-dominance:
      (a) add the new candidate,
      (b) drop existing members it dominates,
      (c) discard it if any existing member dominates it,
      (d) if the archive overflows REFINEMENT_POPULATION_MAX, evict the member
          with the lowest hypervolume contribution.
    """
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
    # Drop any prior copy of the same script before re-inserting.
    population = [
        p for p in list(state.get("refinement_population", []) or [])
        if p.get("script_sha256") != script_hash
    ]
    new_vec = _pareto_objective_vector(metrics_dict)

    # (c) If an existing member dominates the newcomer, it adds nothing — discard.
    for member in population:
        if _dominates(_pareto_objective_vector(member.get("metrics", {})), new_vec):
            state["refinement_population"] = population
            return

    # (b) Remove every existing member the newcomer dominates.
    population = [
        member for member in population
        if not _dominates(new_vec, _pareto_objective_vector(member.get("metrics", {})))
    ]
    # (a) Add the new candidate.
    population.append(entry)

    # (d) Evict the least-contributing member while the archive overflows.
    while len(population) > config.REFINEMENT_POPULATION_MAX:
        evict = _least_hypervolume_contributor(population)
        population.pop(evict)

    state["refinement_population"] = population


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
    generation_fails     = int(tool_context.state.get("generation_fail_count", 0))

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
    smoke = build_smoke_diagnostics(
        debug_result.stdout,
        debug_result.duration_ms,
        context=f"phase2 debug predict outer={n} inner={m}",
    )
    smoke_record = {
        "metrics": smoke.get("metrics"),
        "score": smoke.get("score"),
        "diagnostics": {
            "probe_metrics": smoke.get("probe_metrics"),
            "calibration_stats": smoke.get("calibration_stats"),
            "threshold_curve": smoke.get("threshold_curve"),
            "epoch_logs": smoke.get("epoch_logs"),
            "early_collapse": smoke.get("early_collapse"),
        },
    }
    tool_context.state["latest_smoke_run"] = smoke_record
    if debug_result.returncode != 0:
        logger.warning(
            "Debug pre-check failed (outer=%d, inner=%d, rc=%d) — skipping full run",
            n, m, debug_result.returncode,
        )
        m_next = m + 1
        tool_context.state["inner_iteration"] = m_next
        generation_fails += 1
        tool_context.state["generation_fail_count"] = generation_fails
        if generation_fails >= config.GENERATION_FAIL_MAX:
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
            "generation_fail_count": generation_fails,
            "metrics":         None,
            "smoke_metrics":    smoke_record["metrics"],
            "smoke_score":      smoke_record["score"],
            "smoke_diagnostics": smoke_record["diagnostics"],
            "full_run_executed": False,
            "full_run_reason":  "smoke_check_failed",
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

    # ---- predictive early-abort (KompeteAI): prune EGREGIOUS variants from the
    # cheap micro-run before paying for the full run. The smoke run already prints
    # METRICS on 1 epoch / 5% data; we parse them and only abort when they are
    # *unambiguously* bad. This is intentionally conservative: a 1-epoch/5%-data run
    # is noisy, so the gates (config.DEBUG_PREDICT_*) are far looser than acceptance,
    # and missing/degenerate metrics never prune — borderline candidates get the full run.
    debug_metrics = smoke_record["metrics"]
    if debug_metrics is not None and (
        float(debug_metrics.get("overkill_rate", 0.0)) > config.DEBUG_PREDICT_OVERKILL_MAX
        or float(debug_metrics.get("ng_recall", 1.0)) < config.DEBUG_PREDICT_NG_RECALL_MIN
    ):
        logger.warning(
            "Debug predict pruned (outer=%d, inner=%d): micro-run overkill=%.3f "
            "ng_recall=%.3f — skipping full run",
            n, m, debug_metrics["overkill_rate"], debug_metrics["ng_recall"],
        )
        m_next = m + 1
        tool_context.state["inner_iteration"] = m_next
        generation_fails += 1
        tool_context.state["generation_fail_count"] = generation_fails
        if generation_fails >= config.GENERATION_FAIL_MAX:
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
            "failure_reason":  "debug_predicted_low_utility",
            "generation_fail_count": generation_fails,
            "new_score":       0.0,
            "new_overkill":    1.0,
            "current_best_score":   float(tool_context.state.get("current_best_score", 0.0)),
            "current_best_overkill": float(tool_context.state.get("best_overkill_rate", 1.0)),
            "no_improve_count":   no_improve,
            "metrics":         None,
            "smoke_metrics":    smoke_record["metrics"],
            "smoke_score":      smoke_record["score"],
            "smoke_diagnostics": smoke_record["diagnostics"],
            "full_run_executed": False,
            "full_run_reason":  "smoke_pruned_egregious",
            "stdout_tail":     debug_result.stdout[-3000:],
            "stderr_tail":     debug_result.stderr[-1000:],
        }
        save_checkpoint(config.ckpt_refinement(n, m), attempt_data)
        return (
            f"DEBUG PREDICT PRUNED (outer={n}, inner={m}): accelerated smoke run "
            f"already shows overkill={debug_metrics['overkill_rate']:.3f} "
            f"(max {config.DEBUG_PREDICT_OVERKILL_MAX}) / "
            f"ng_recall={debug_metrics['ng_recall']:.3f} "
            f"(min {config.DEBUG_PREDICT_NG_RECALL_MIN}); full run skipped. "
            f"inner_iteration now {m_next}, no_improve_count now {no_improve}."
        )

    # ---- curve-abort: extrapolate the SHORT debug learning-curves and prune the
    # full run ONLY when BOTH projected finals are CONFIDENTLY worse than the
    # current best. The debug micro-run now emits config.CURVE_ABORT_DEBUG_EPOCHS
    # epochs on 5% data (same 120s cap), so we have per-epoch val_ng_recall AND
    # val_overkill curves. We fit a saturating power-law to each and prune only
    # when the projection is trustworthy (fit_quality >= CURVE_ABORT_MIN_FIT) and
    # confidently worse than the best seen so far: ng_recall below best by
    # CURVE_ABORT_MARGIN, AND overkill above best by CURVE_ABORT_OVERKILL_MARGIN
    # (looser, since overkill is noisier on 5%-data micro-runs). AND logic: a run
    # is only hopeless when BOTH trajectories project bad — if EITHER still
    # projects healthy, threshold tuning may rescue it (overkill is especially
    # threshold-fixable), so it falls through to the full run. Safe-by-default:
    # too few / poor-fit points, or no established baseline (best NG recall ~0 /
    # best overkill ~1.0 before any successful full run), never prune.
    # A 5%-data short curve under-shoots the full-data plateau, hence the margins.
    epoch_curve = (smoke_record.get("diagnostics") or {}).get("epoch_logs") or []
    ng_recall_series = [
        e.get("val_ng_recall")
        for e in epoch_curve
        if isinstance(e, dict) and e.get("val_ng_recall") is not None
    ]
    projected_ng_recall, curve_fit = project_power_law(ng_recall_series)
    best_ng_recall = max(0.0, 1.0 - current_best_miss)
    ng_recall_curve_bad = (
        projected_ng_recall is not None
        and curve_fit >= config.CURVE_ABORT_MIN_FIT
        and projected_ng_recall < best_ng_recall - config.CURVE_ABORT_MARGIN
    )

    # Overkill is a rate where LOWER is better, so a doomed trajectory projects a
    # final overkill HIGHER than the best so far. Mirror the ng_recall projection
    # with its own (looser) margin. Safe-by-default mirrors ng_recall: too few
    # points, a poor fit, or no baseline (best_overkill_rate ~1.0, which the
    # clamped-to-[0,1] projection can never exceed by the margin) -> not bad.
    overkill_series = [
        e.get("val_overkill")
        for e in epoch_curve
        if isinstance(e, dict) and e.get("val_overkill") is not None
    ]
    projected_overkill, overkill_fit = project_power_law(overkill_series)
    overkill_curve_bad = (
        projected_overkill is not None
        and overkill_fit >= config.CURVE_ABORT_MIN_FIT
        and projected_overkill > current_best_overkill + config.CURVE_ABORT_OVERKILL_MARGIN
    )

    if ng_recall_curve_bad and overkill_curve_bad:
        logger.warning(
            "Curve abort pruned (outer=%d, inner=%d): projected ng_recall=%.3f "
            "(fit=%.2f) below best ng_recall=%.3f by > margin %.2f AND projected "
            "overkill=%.3f (fit=%.2f) above best overkill=%.3f by > margin %.2f "
            "— skipping full run",
            n, m, projected_ng_recall, curve_fit, best_ng_recall,
            config.CURVE_ABORT_MARGIN,
            projected_overkill, overkill_fit, current_best_overkill,
            config.CURVE_ABORT_OVERKILL_MARGIN,
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
            "failure_reason":  "curve_abort_projected_low_utility",
            "new_score":       0.0,
            "new_overkill":    1.0,
            "current_best_score":   float(tool_context.state.get("current_best_score", 0.0)),
            "current_best_overkill": float(tool_context.state.get("best_overkill_rate", 1.0)),
            "no_improve_count":   no_improve,
            "metrics":         None,
            "smoke_metrics":    smoke_record["metrics"],
            "smoke_score":      smoke_record["score"],
            "smoke_diagnostics": smoke_record["diagnostics"],
            "projected_ng_recall": round(float(projected_ng_recall), 4),
            "curve_fit_quality":   round(float(curve_fit), 4),
            "projected_overkill":  round(float(projected_overkill), 4),
            "overkill_curve_fit_quality": round(float(overkill_fit), 4),
            "full_run_executed": False,
            "full_run_reason":  "curve_abort_projected_low",
            "stdout_tail":     debug_result.stdout[-3000:],
            "stderr_tail":     debug_result.stderr[-1000:],
        }
        save_checkpoint(config.ckpt_refinement(n, m), attempt_data)
        return (
            f"CURVE ABORT PRUNED (outer={n}, inner={m}): debug learning-curves "
            f"project final ng_recall={projected_ng_recall:.3f} "
            f"(fit_quality={curve_fit:.2f} >= {config.CURVE_ABORT_MIN_FIT}), "
            f"confidently below best ng_recall={best_ng_recall:.3f} by more than "
            f"margin {config.CURVE_ABORT_MARGIN}, AND final overkill="
            f"{projected_overkill:.3f} (fit_quality={overkill_fit:.2f} >= "
            f"{config.CURVE_ABORT_MIN_FIT}), confidently above best overkill="
            f"{current_best_overkill:.3f} by more than margin "
            f"{config.CURVE_ABORT_OVERKILL_MARGIN}; full run skipped. "
            f"inner_iteration now {m_next}, no_improve_count now {no_improve}."
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
        _state_pop(tool_context.state, "latest_probe_metrics")
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

    # Probe rejection is an EARLY-ABORT signal: it only applies when the run
    # produced no valid final METRICS (script aborted at probe, or its output
    # was degenerate). A completed full run with guarded metrics is always
    # judged on those metrics — a weak epoch-5 probe must not retroactively
    # discard 20 epochs of real training evidence.
    probe_rejected = (
        result.returncode == 0
        and probe_rejection_reason is not None
        and metrics is None
    )
    run_ok      = result.returncode == 0 and metrics is not None and not probe_rejected
    new_score   = float(metrics.ng_recall)    if metrics else 0.0
    new_overkill = float(metrics.overkill_rate) if metrics else 1.0

    if run_ok and metric_guard.is_no_signal_metrics(metrics):
        m_next = m + 1
        tool_context.state["inner_iteration"] = m_next
        tool_context.state["stop_outer_loop"] = True
        tool_context.state["phase2_abort_reason"] = "no_cross_lot_signal"

        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        error_analysis_path = config.ckpt_error_analysis(n, m)
        error_analysis_checkpoint = {
            "outer_iteration": n,
            "inner_iteration": m,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "metrics": metrics_to_dict(metrics),
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
            "improved":        False,
            "failure_reason":  "no_cross_lot_signal",
            "new_score":       new_score,
            "new_overkill":    new_overkill,
            "current_best_score":   current_best,
            "current_best_overkill": current_best_overkill,
            "no_improve_count":   no_improve,
            "metrics":         metrics_to_dict(metrics),
            "smoke_metrics":    smoke_record["metrics"],
            "smoke_score":      smoke_record["score"],
            "smoke_diagnostics": smoke_record["diagnostics"],
            "full_run_executed": True,
            "full_run_reason":  "no_cross_lot_signal_abort",
            "error_analysis_checkpoint": str(error_analysis_path),
            "error_analysis_available": error_analysis.get("available", False),
            "threshold_curve": threshold_curve,
            "probe_metrics": probe_metrics,
            "probe_rejection_reason": probe_rejection_reason,
            "stdout_tail":     result.stdout[-3000:],
            "stderr_tail":     result.stderr[-1000:],
        }
        save_checkpoint(config.ckpt_refinement(n, m), attempt_data)
        _save_best_pipeline(tool_context.state)
        tool_context.actions.escalate = True
        return (
            f"[outer={n}, inner={m}] NO_CROSS_LOT_SIGNAL: held-out roc_auc="
            f"{metrics.roc_auc:.3f} <= {metric_guard.NO_SIGNAL_ROC_AUC_MAX:.2f}; "
            "no cross-lot signal — needs new strategy. stop_outer_loop=True set; "
            "refinement halted before accepting or refining this candidate further."
        )

    # Clear instrumentation flag once the script successfully emits per-sample evidence
    if error_analysis.get("available"):
        _state_pop(tool_context.state, "error_analysis_instrumentation_required")
        _state_pop(tool_context.state, "error_analysis_repair_attempted")
        _state_pop(tool_context.state, "error_analysis_blocked")

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
    selection_evaluation = None
    if run_ok:
        improved, selected_metrics, selection_evaluation = _confirm_improvement_with_selection_average(
            script=script,
            metrics=metrics,
            current_metrics=current_metrics,
            initially_improved=improved,
        )
        if selection_evaluation is not None:
            if selection_evaluation.get("mode") == "board_grouped_cv":
                tool_context.state["latest_cv_evaluation"] = selection_evaluation
                _state_pop(tool_context.state, "latest_multiseed_confirmation")
            else:
                tool_context.state["latest_multiseed_confirmation"] = selection_evaluation
                _state_pop(tool_context.state, "latest_cv_evaluation")
            if selection_evaluation.get("status") == "success":
                metrics = _metrics_from_dict(selected_metrics)
                new_score = float(metrics.ng_recall)
                new_overkill = float(metrics.overkill_rate)
        else:
            _state_pop(tool_context.state, "latest_multiseed_confirmation")
            _state_pop(tool_context.state, "latest_cv_evaluation")
    else:
        _state_pop(tool_context.state, "latest_multiseed_confirmation")
        _state_pop(tool_context.state, "latest_cv_evaluation")

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
    # The prediction contract is a DIAGNOSTIC signal for the planner (it shows
    # whether the diagnosis hypothesis held), never a veto on a real metric
    # improvement. The previous veto ("prediction failed and below relaxed
    # acceptance => improved = False") structurally disabled hill-climbing:
    # the diagnosis LLM writes optimistic targets, so every incremental gain
    # (e.g. recall 0.3 -> 0.6) was rejected and the best score stayed at 0.

    # Capture the previous best script BEFORE the improvement block can overwrite
    # it below; the persistent KB diff is prev_best -> current.
    prev_best_script = tool_context.state.get("best_pipeline_script", "") or ""

    if improved:
        tool_context.state["current_best_score"]   = new_score
        tool_context.state["best_overkill_rate"]   = new_overkill
        tool_context.state["best_accuracy"]         = float(metrics.accuracy)
        tool_context.state["best_f1"]               = float(metrics.f1)
        tool_context.state["best_miss_rate"]        = float(metrics.miss_rate)
        tool_context.state["best_roc_auc"]          = float(metrics.roc_auc)
        tool_context.state["best_prob_gap"]         = float(metrics.prob_gap)
        tool_context.state["best_selection_evaluation"] = selection_evaluation
        tool_context.state["best_pipeline_script"] = script
        tool_context.state["no_improve_count"]     = 0
        tool_context.state["generation_fail_count"] = 0
        no_improve = 0
        generation_fails = 0
        logger.info(
            "New best: ng_recall=%.4f overkill=%.4f (was ng_recall=%.4f overkill=%.4f)",
            new_score, new_overkill, current_best, current_best_overkill,
        )
    elif probe_rejected or not run_ok:
        # Script never trained — count as generation failure, not training failure.
        generation_fails += 1
        tool_context.state["generation_fail_count"] = generation_fails
        if generation_fails >= config.GENERATION_FAIL_MAX:
            no_improve += 1
            tool_context.state["no_improve_count"] = no_improve
    else:
        # Full training ran but didn't improve — counts against patience.
        generation_fails = 0
        tool_context.state["generation_fail_count"] = generation_fails
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
        "smoke_metrics":    smoke_record["metrics"],
        "smoke_score":      smoke_record["score"],
        "smoke_diagnostics": smoke_record["diagnostics"],
        "full_run_executed": True,
        "full_run_reason":  "full_run_after_smoke",
        "error_analysis_checkpoint": str(error_analysis_path),
        "error_analysis_available": error_analysis.get("available", False),
        "threshold_curve": threshold_curve,
        "probe_metrics": probe_metrics,
        "probe_rejection_reason": probe_rejection_reason,
        "prediction_verification": prediction_verification,
        AVERAGED_EVALUATION_KEY: selection_evaluation,
        "cv_evaluation": (
            selection_evaluation
            if isinstance(selection_evaluation, dict)
            and selection_evaluation.get("mode") == "board_grouped_cv"
            else None
        ),
        "refinement_population_count": len(tool_context.state.get("refinement_population", []) or []),
        "stdout_tail":     result.stdout[-3000:],
        "stderr_tail":     result.stderr[-1000:],
    }
    save_checkpoint(config.ckpt_refinement(n, m), attempt_data)

    # Append to tried_approaches memory (P4)
    prior_tried = list(tool_context.state.get("tried_approaches", []) or [])
    tried = list(prior_tried)
    plan = tool_context.state.get("refinement_plan") or {}
    selected_strategy = tool_context.state.get("selected_refinement_strategy", "")
    result_dict = {
        "ng_recall": round(new_score, 4),
        "miss_rate": round(float(metrics.miss_rate), 4) if metrics else 1.0,
        "overkill": round(new_overkill, 4),
        "accuracy": round(float(metrics.accuracy), 4) if metrics else 0.0,
        "improved": improved,
    }
    selected_fingerprint = tool_context.state.get("selected_strategy_fingerprint") or {}
    target_component = _normalise_target_component(plan.get("target_component", "unknown"))
    if selected_fingerprint:
        selected_fingerprint = dict(selected_fingerprint)
        selected_fingerprint["target_component"] = _normalise_target_component(
            selected_fingerprint.get("target_component", target_component)
        )
    tried.append({
        "outer": n, "inner": m,
        "target_component": target_component,
        "changes_summary": plan.get("changes_summary", ""),
        "selected_strategy": selected_strategy,
        "strategy_fingerprint": selected_fingerprint,
        "attempt_label": _attempt_label(target_component, selected_fingerprint),
        "result": result_dict,
        "prediction_verification": prediction_verification,
        "failure_reason": failure_reason,
    })
    tool_context.state["tried_approaches"] = tried
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(config.CKPT_TRIED_APPROACHES, {"tried_approaches": tried})

    # Persistent AOI knowledge base (MLEvolve "Experience-Driven Global Memory"):
    # bounded cross-run memory. Each record is:
    #   {plan, code_diff, metrics, tags, target_component, mechanism_class}
    # - code_diff is a TRUNCATED unified diff (prev best -> current); never the
    #   full script.
    # - tags is a categorical list: [failure_mode (if any), "improved"|"regressed"].
    # - target_component/mechanism_class come from the selected strategy fingerprint
    #   and key the dedup signature alongside tags.
    # Load the existing list first and write the whole list back — never overwrite.
    # The append is skipped when the new record's (tags, target_component,
    # mechanism_class) signature matches a recent record, and a FIFO cap keeps only
    # the most recent PERSISTENT_KB_MAX_RECORDS so the memory cannot grow unbounded.
    # Wrapped so a KB write failure never raises into the evaluator hot path.
    # Guard: skip KB write for stub scripts (no real training output) or when
    # metrics are absent — these would poison the KB with fake results.
    _script_is_stub = len((script or "").strip().splitlines()) < 10
    _metrics_are_real = metrics is not None and getattr(metrics, "roc_auc", 0) > 0
    if _script_is_stub or not _metrics_are_real:
        logger.info(
            "Persistent AOI KB write skipped: script is stub (%s) or metrics absent/zero roc_auc (%s).",
            _script_is_stub, not _metrics_are_real,
        )
    else:
        try:
            kb_records = []
            if checkpoint_exists(config.CKPT_PERSISTENT_KB):
                existing_kb = load_checkpoint(config.CKPT_PERSISTENT_KB)
                if isinstance(existing_kb, list):
                    kb_records = list(existing_kb)

            # failure_mode lives under failure_classification in diagnosis_brief
            # (primary) or diagnosis_report (mirror). Empty-safe.
            failure_mode = ""
            for source_key in ("diagnosis_brief", "diagnosis_report"):
                blob = tool_context.state.get(source_key) or {}
                if isinstance(blob, dict):
                    classification = blob.get("failure_classification") or {}
                    if isinstance(classification, dict):
                        failure_mode = (classification.get("failure_mode") or "").strip()
                        if failure_mode:
                            break

            kb_tags = []
            if failure_mode:
                kb_tags.append(failure_mode)
            kb_tags.append("improved" if improved else "regressed")

            kb_metrics = {
                "ng_recall":     result_dict["ng_recall"],
                "miss_rate":     result_dict["miss_rate"],
                "overkill_rate": result_dict["overkill"],
                "accuracy":      result_dict["accuracy"],
                "improved":      result_dict["improved"],
            }

            kb_target_component = (
                selected_fingerprint.get("target_component") or target_component
            )
            kb_mechanism_class = (selected_fingerprint.get("mechanism_class") or "unknown")
            kb_code_diff = make_code_diff(prev_best_script, script)
            tried_for_harmful = tried if not improved else prior_tried

            new_kb_record = {
                "plan": selected_strategy,
                "code_diff": kb_code_diff,
                "metrics": kb_metrics,
                "tags": kb_tags,
                "target_component": kb_target_component,
                "mechanism_class": kb_mechanism_class,
                "helpful_mechanisms": _helpful_mechanisms_from_state(
                    tool_context.state,
                    improved=improved,
                    selected_fingerprint=selected_fingerprint,
                    code_diff=kb_code_diff,
                ),
                "harmful_mechanisms": _harmful_mechanisms_from_tried_approaches(
                    tried_for_harmful,
                    n,
                ),
            }

            # Skip the append when an identical (tags, target_component, mechanism_class)
            # signature already appears among the most recent records — repeated identical
            # attempts must not flood the cross-run memory.
            new_signature = _kb_dedup_signature(new_kb_record)
            recent_window = kb_records[-PERSISTENT_KB_DEDUP_RECENT_WINDOW:]
            is_recent_duplicate = any(
                _kb_dedup_signature(rec) == new_signature for rec in recent_window
            )
            if is_recent_duplicate:
                logger.info(
                    "Persistent AOI KB append skipped: signature %s matches a recent record.",
                    new_signature,
                )
            else:
                kb_records.append(new_kb_record)
                # FIFO cap — keep only the most recent PERSISTENT_KB_MAX_RECORDS.
                if len(kb_records) > PERSISTENT_KB_MAX_RECORDS:
                    kb_records = kb_records[-PERSISTENT_KB_MAX_RECORDS:]
                save_checkpoint(config.CKPT_PERSISTENT_KB, kb_records)
        except Exception:
            logger.exception("Persistent AOI KB write failed; continuing.")

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
    cv_summary = _format_cv_evaluation(selection_evaluation)
    if cv_summary:
        summary_prefix = f"{summary_prefix}\n{cv_summary}"

    # ---- Priority 1: no acceptable path remains ----
    # Below relaxed acceptance, the historical behavior was to restart outer
    # diagnosis indefinitely after stagnation. Once the current ablation evidence
    # contains no relaxed-acceptable result and the existing below-relaxed
    # no-improve threshold has been reached, stop the outer loop instead.
    if loop_guard.should_stop_when_no_acceptable_path_remains(tool_context.state):
        tool_context.state["stop_outer_loop"] = True
        tool_context.state["phase2_abort_reason"] = "no_acceptable_path_remaining"
        _save_best_pipeline(tool_context.state)
        tool_context.actions.escalate = True
        logger.info(
            "Early-stop triggered: no acceptable path remains after current ablation evidence."
        )
        return (
            f"{summary_prefix}\n"
            "EARLY_STOP (no acceptable path remains): current best is below relaxed "
            "acceptance, no ablation result satisfies relaxed acceptance, and the "
            f"below-relaxed no-improve streak reached "
            f"{config.INNER_STAGNATION_MAX_UNCONSTRAINED}. stop_outer_loop=True set. "
            "Escalating inner loop; outer loop exits on next ablation_flag_checker call."
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
    warm_restart_failed = (
        not improved
        and warm_restart.is_warm_restart_fingerprint(
            tool_context.state.get("selected_strategy_fingerprint")
        )
    )
    stagnated = loop_guard.should_restart_inner_for_stagnation(tool_context.state)
    if stagnated and not warm_restart_failed:
        warm_restart_status = warm_restart.arm_warm_restart_if_eligible(tool_context.state)
        if warm_restart_status.startswith("WARM_RESTART_ARMED"):
            tool_context.state["no_improve_count"] = 0
            _save_best_pipeline(tool_context.state)
            logger.info(
                "Inner stagnation hit below relaxed acceptance — armed warm restart."
            )
            return (
                f"{summary_prefix}\n"
                f"{warm_restart_status}\n"
                "CONTINUE: warm-restart refinement armed; inner loop will try it "
                "before ideation or approach abandonment."
            )

    if stagnated or warm_restart_failed:
        n_next = n + 1
        # Cross-branch fusion arming (MCGS Evolution/Fusion): when the archive holds
        # >=2 diverse members, arm a fusion attempt for the upcoming outer cycle so
        # the strategy gate fuses their winning blocks instead of refining a single
        # chain. Must run BEFORE the counters below are reset — should_trigger_fusion
        # re-checks the stagnation signal, which depends on no_improve_count.
        fusion_status = fusion.arm_fusion_if_eligible(tool_context.state, target_outer=n_next)
        tool_context.state["outer_iteration"] = n_next
        tool_context.state["inner_iteration"] = 0
        tool_context.state["no_improve_count"] = 0
        tool_context.state["force_fresh_ablation"] = True
        _clear_ablation_state(tool_context.state)
        tool_context.state.pop("reflexion_memo_text", None)
        tool_context.state.pop("reflexion_memo", None)
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
        warm_restart_line = (
            "WARM_RESTART_FAILED: bounded warm-restart attempt did not improve; "
            "falling through to existing ideation/restart path.\n"
            if warm_restart_failed else ""
        )
        return (
            f"{summary_prefix}\n"
            f"{warm_restart_line}"
            f"INNER_STAGNATION: no improvement for "
            f"{config.INNER_STAGNATION_MAX_UNCONSTRAINED} below-relaxed attempts. "
            f"outer_iteration advanced to {n_next}, inner_iteration reset to 0, "
            "and ablation state cleared. Escalating inner loop; outer loop continues.\n"
            f"{ideation_status}\n{fusion_status}"
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
  - For promising candidates that pass the smoke/full-run gate, runs board-grouped
    3-fold CV instead of separate multiseed confirmation. The returned summary
    includes per-fold validation metrics plus aggregate worst-fold NG recall /
    miss rate and mean overkill / accuracy.
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
2. Any BOARD_GROUPED_CV fold metrics included in the tool output
3. Whether the script improved the best score
4. Current loop counters: outer_iteration, inner_iteration, no_improve_count
5. The exit decision (CONTINUE / INNER_CAP / EARLY_STOP) as returned by the tool

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
