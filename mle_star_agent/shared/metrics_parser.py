import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from mle_star_agent.shared.labels import normalize_label

logger = logging.getLogger(__name__)

METRICS_LABEL = "METRICS"
PROBE_METRICS_LABEL = "PROBE_METRICS"
EPOCH_LOG_LABEL = "EPOCH_LOG"
CALIBRATION_STATS_LABEL = "CALIBRATION_STATS"
THRESHOLD_CURVE_LABEL = "THRESHOLD_CURVE"
PREDICTIONS_LABEL = "PREDICTIONS"
ERROR_ANALYSIS_LABEL = "ERROR_ANALYSIS"

REQUIRED_GENERATED_SCRIPT_MARKERS = (
    f"{PROBE_METRICS_LABEL}:",
    f"{EPOCH_LOG_LABEL}:",
    f"{METRICS_LABEL}:",
    f"{CALIBRATION_STATS_LABEL}:",
    f"{THRESHOLD_CURVE_LABEL}:",
    f"{PREDICTIONS_LABEL}:",
)


@dataclass
class AOIMetrics:
    accuracy: float
    ng_recall: float
    miss_rate: float
    overkill_rate: float
    f1: float
    avg_latency_ms: float
    threshold: float
    ng_count: int
    g_count: int
    tp: int
    tn: int
    fp: int
    fn: int
    roc_auc: float = 0.0
    prob_gap: float = 0.0


def parse_metrics(stdout: str) -> Optional[AOIMetrics]:
    # Use the depth-tracking extractor so nested braces in the JSON don't cause truncation
    raw = _extract_json_block(stdout, METRICS_LABEL)
    if not isinstance(raw, dict):
        return None

    tp = int(raw.get("tp", 0))
    tn = int(raw.get("tn", 0))
    fp = int(raw.get("fp", 0))
    fn = int(raw.get("fn", 0))

    if tp + fn == 0:
        logger.warning("TP+FN=0: no NG samples in split; setting ng_recall=1.0, miss_rate=0.0")
        ng_recall = 1.0
        miss_rate = 0.0
    else:
        ng_recall = tp / (tp + fn)
        miss_rate = fn / (tp + fn)

    if tn + fp == 0:
        logger.warning("TN+FP=0: no G samples in split; setting overkill_rate=0.0")
        overkill_rate = 0.0
    else:
        overkill_rate = fp / (tn + fp)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * ng_recall) / (precision + ng_recall) if (precision + ng_recall) > 0 else 0.0

    return AOIMetrics(
        accuracy=accuracy,
        ng_recall=ng_recall,
        miss_rate=miss_rate,
        overkill_rate=overkill_rate,
        f1=f1,
        avg_latency_ms=float(raw.get("avg_latency_ms", 0.0)),
        threshold=float(raw.get("threshold", 0.5)),
        ng_count=int(raw.get("ng_count", tp + fn)),
        g_count=int(raw.get("g_count", tn + fp)),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        roc_auc=float(raw.get("roc_auc", 0.0) or 0.0),
        prob_gap=float(raw.get("prob_gap", 0.0) or 0.0),
    )


def metrics_to_dict(m: AOIMetrics) -> dict:
    return {
        "accuracy": m.accuracy,
        "ng_recall": m.ng_recall,
        "miss_rate": m.miss_rate,
        "overkill_rate": m.overkill_rate,
        "f1": m.f1,
        "avg_latency_ms": m.avg_latency_ms,
        "threshold": m.threshold,
        "ng_count": m.ng_count,
        "g_count": m.g_count,
        "tp": m.tp,
        "tn": m.tn,
        "fp": m.fp,
        "fn": m.fn,
        "roc_auc": m.roc_auc,
        "prob_gap": m.prob_gap,
    }


def _extract_json_block(stdout: str, label: str) -> Any:
    match = re.search(rf"(?m)^\s*{re.escape(label)}:\s*", stdout)
    if not match:
        return None

    start = match.end()
    while start < len(stdout) and stdout[start].isspace():
        start += 1

    if start >= len(stdout) or stdout[start] not in "{[":
        return None

    opener = stdout[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for pos in range(start, len(stdout)):
        ch = stdout[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stdout[start:pos + 1])
                except json.JSONDecodeError:
                    return None

    return None


def parse_probe_metrics(stdout: str) -> Optional[dict]:
    """Parse PROBE_METRICS: {...} from generated script stdout."""
    raw = _extract_json_block(stdout, PROBE_METRICS_LABEL)
    if not isinstance(raw, dict):
        return None

    g_prob_mean = _float_or_none(
        raw.get("g_prob_mean", raw.get("G_prob_mean", raw.get("g_mean")))
    )
    ng_prob_mean = _float_or_none(
        raw.get("ng_prob_mean", raw.get("NG_prob_mean", raw.get("ng_mean")))
    )
    probability_gap = _float_or_none(
        raw.get("probability_gap", raw.get("separability_gap", raw.get("margin")))
    )
    if probability_gap is None and g_prob_mean is not None and ng_prob_mean is not None:
        probability_gap = ng_prob_mean - g_prob_mean

    parsed = dict(raw)
    parsed.update({
        "source": PROBE_METRICS_LABEL,
        "g_prob_mean": g_prob_mean,
        "ng_prob_mean": ng_prob_mean,
        "probability_gap": probability_gap,
        "ng_recall": _float_or_none(raw.get("ng_recall", raw.get("recall"))),
        "miss_rate": _float_or_none(raw.get("miss_rate")),
        "overkill_rate": _float_or_none(raw.get("overkill_rate", raw.get("overkill"))),
        "accuracy": _float_or_none(raw.get("accuracy")),
    })
    return parsed


def _normalise_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    # Map to canonical G/NG using the configured PASS_LABELS / FAIL_LABELS.
    # Non-strict: model-output parsing is best-effort, so an unrecognised value
    # falls back to its upper-cased raw form (unchanged historical behaviour)
    # rather than raising.
    canon = normalize_label(value, strict=False)
    if canon is not None:
        return canon
    return str(value).strip().upper()


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_prediction(raw: Any, index: int) -> dict:
    if not isinstance(raw, dict):
        raw = {"raw": raw}

    sample_id = (
        raw.get("sample_id")
        or raw.get("id")
        or raw.get("image_id")
        or raw.get("pair_id")
        or raw.get("img_l")
        or raw.get("path")
        or f"sample_{index}"
    )
    true_label = _normalise_label(raw.get("true_label", raw.get("label", raw.get("y_true"))))
    predicted_label = _normalise_label(raw.get("predicted_label", raw.get("prediction", raw.get("y_pred"))))
    ng_probability = _float_or_none(
        raw.get("ng_probability", raw.get("ng_prob", raw.get("probability", raw.get("score"))))
    )
    threshold = _float_or_none(raw.get("threshold"))
    lot = raw.get("lot")
    if not lot:
        sample_text = str(sample_id)
        parts = sample_text.split("/")
        lot = parts[0] if len(parts) > 1 else None
    if not lot:
        for key in ("img_l", "img_r", "path"):
            value = raw.get(key)
            if value:
                lot = os.path.basename(os.path.dirname(str(value)))
                break

    error_type = raw.get("error_type")
    if not error_type and true_label in {"G", "NG"} and predicted_label in {"G", "NG"}:
        if true_label == "G" and predicted_label == "NG":
            error_type = "FP"
        elif true_label == "NG" and predicted_label == "G":
            error_type = "FN"
        elif true_label == "NG" and predicted_label == "NG":
            error_type = "TP"
        elif true_label == "G" and predicted_label == "G":
            error_type = "TN"

    return {
        "sample_id": str(sample_id),
        "img_l": raw.get("img_l"),
        "img_r": raw.get("img_r"),
        "lot": str(lot) if lot else None,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "ng_probability": ng_probability,
        "threshold": threshold,
        "error_type": str(error_type).upper() if error_type else None,
    }


def _probability_summary(predictions: list[dict]) -> dict:
    summary = {}
    for label in ("G", "NG"):
        probs = [
            p["ng_probability"]
            for p in predictions
            if p.get("true_label") == label and p.get("ng_probability") is not None
        ]
        if probs:
            summary[label] = {
                "count": len(probs),
                "min": min(probs),
                "max": max(probs),
                "mean": sum(probs) / len(probs),
            }
        else:
            summary[label] = {"count": 0, "min": None, "max": None, "mean": None}
    return summary


def _per_lot_summary(predictions: list[dict]) -> dict:
    lots: dict[str, dict] = {}
    for prediction in predictions:
        lot = prediction.get("lot") or "unknown"
        entry = lots.setdefault(lot, {
            "total": 0,
            "g_count": 0,
            "ng_count": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "overkill_rate": 0.0,
            "miss_rate": 0.0,
        })
        entry["total"] += 1
        if prediction.get("true_label") == "G":
            entry["g_count"] += 1
        elif prediction.get("true_label") == "NG":
            entry["ng_count"] += 1

        error_type = prediction.get("error_type")
        if error_type in {"TP", "TN", "FP", "FN"}:
            entry[error_type.lower()] += 1

    for entry in lots.values():
        g_total = entry["tn"] + entry["fp"]
        ng_total = entry["tp"] + entry["fn"]
        entry["overkill_rate"] = entry["fp"] / g_total if g_total else 0.0
        entry["miss_rate"] = entry["fn"] / ng_total if ng_total else 0.0
    return lots


def _metrics_consistency(metrics: Optional[AOIMetrics], fp_count: int, fn_count: int) -> dict:
    if metrics is None:
        return {
            "expected_fp": None,
            "expected_fn": None,
            "parsed_fp": fp_count,
            "parsed_fn": fn_count,
            "matches_metrics": None,
        }
    return {
        "expected_fp": metrics.fp,
        "expected_fn": metrics.fn,
        "parsed_fp": fp_count,
        "parsed_fn": fn_count,
        "matches_metrics": metrics.fp == fp_count and metrics.fn == fn_count,
    }


def parse_error_analysis(stdout: str, metrics: Optional[AOIMetrics] = None) -> dict:
    """Parse deterministic per-sample error evidence from generated script stdout."""
    raw_error_analysis = _extract_json_block(stdout, ERROR_ANALYSIS_LABEL)
    raw_predictions = _extract_json_block(stdout, PREDICTIONS_LABEL)

    source = None
    predictions_raw = None
    if isinstance(raw_predictions, list):
        source = PREDICTIONS_LABEL
        predictions_raw = raw_predictions
    elif isinstance(raw_error_analysis, dict):
        source = ERROR_ANALYSIS_LABEL
        predictions_raw = raw_error_analysis.get("predictions")

    if isinstance(predictions_raw, list):
        predictions = [
            _normalise_prediction(prediction, index)
            for index, prediction in enumerate(predictions_raw)
        ]
        fp_samples = [p for p in predictions if p.get("error_type") == "FP"]
        fn_samples = [p for p in predictions if p.get("error_type") == "FN"]
        return {
            "available": True,
            "source": source,
            "predictions_count": len(predictions),
            "fp_count": len(fp_samples),
            "fn_count": len(fn_samples),
            "fp_samples": fp_samples,
            "fn_samples": fn_samples,
            "probability_summary": _probability_summary(predictions),
            "per_lot": _per_lot_summary(predictions),
            "metrics_consistency": _metrics_consistency(metrics, len(fp_samples), len(fn_samples)),
        }

    if isinstance(raw_error_analysis, dict):
        fp_samples = [
            _normalise_prediction(sample, index)
            for index, sample in enumerate(raw_error_analysis.get("fp_samples", []))
        ]
        fn_samples = [
            _normalise_prediction(sample, index)
            for index, sample in enumerate(raw_error_analysis.get("fn_samples", []))
        ]
        return {
            "available": True,
            "source": "ERROR_ANALYSIS",
            "predictions_count": raw_error_analysis.get("predictions_count"),
            "fp_count": len(fp_samples),
            "fn_count": len(fn_samples),
            "fp_samples": fp_samples,
            "fn_samples": fn_samples,
            "probability_summary": raw_error_analysis.get("probability_summary", {}),
            "per_lot": raw_error_analysis.get(
                "per_lot", _per_lot_summary(fp_samples + fn_samples)
            ),
            "metrics_consistency": _metrics_consistency(metrics, len(fp_samples), len(fn_samples)),
            "notes": raw_error_analysis.get("notes"),
        }

    expected_fp = metrics.fp if metrics else None
    expected_fn = metrics.fn if metrics else None
    return {
        "available": False,
        "source": None,
        "missing_reason": "No ERROR_ANALYSIS or PREDICTIONS block found in stdout.",
        "predictions_count": 0,
        "fp_count": None,
        "fn_count": None,
        "fp_samples": [],
        "fn_samples": [],
        "probability_summary": {},
        "metrics_consistency": {
            "expected_fp": expected_fp,
            "expected_fn": expected_fn,
            "parsed_fp": None,
            "parsed_fn": None,
            "matches_metrics": False if metrics else None,
        },
    }
