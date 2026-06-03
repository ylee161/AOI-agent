"""Deterministic pre-LLM diagnosis scorer (P2+P3).

Moves arithmetic (ablation delta computation, failure classification, variant ranking)
out of the LLM to save 30-60% diagnosis tokens and eliminate narrative misinterpretation.
"""

import json
import logging
import re
from typing import Any, Optional

from mle_star_agent import config
from mle_star_agent.shared.acceptance_scoring import acceptance_distance, metrics_view
from mle_star_agent.shared.metrics_parser import _extract_json_block
from mle_star_agent.shared.small_data_strategy_validator import KNOWN_FAILED_STRATEGY_FINGERPRINTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure mode taxonomy (P2A)
# ---------------------------------------------------------------------------

FAILURE_MODES = {
    "threshold_collapse": "G_prob_mean > 0.50; model predicts NG on most inputs to maximize recall",
    "g_ng_overlap": "G and NG probability distributions overlap; model has no separability",
    "class_imbalance_overfit": "High train recall but low val recall; overfitting to NG class",
    "preprocessing_lot_shift": "Accuracy varies across lot folders; lighting/calibration mismatch",
    "low_capacity_miss": "FN samples are far from threshold; model lacks capacity",
    "full_freeze_underfit": "Full backbone freezing collapsed prob_gap; last-block adaptation is needed",
    "near_acceptance": "Metrics are close to relaxed targets; fine-tuning may suffice",
}


def small_data_strategy_policy(input_modality: str = "stereo") -> dict:
    """Return the diagnosis/planner policy for the 287-sample grouped train split.

    ``input_modality`` defaults to "stereo" (preserves all existing behaviour). When
    "mono", stereo-only deprioritised strategies and L/R paired-augmentation safety text
    are removed.
    """
    is_mono = input_modality == "mono"

    prefer_order = [
        "partial_unfreeze_last_resnet_block_small_head",
        "freeze_early_backbone_layers",
        "freeze_or_partial_freeze_small_head",
        "weight_decay_dropout",
        "aoi_safe_paired_augmentation" if not is_mono else "aoi_safe_augmentation",
        "smaller_regularized_head",
        ("local_patch_evidence" if is_mono
         else "local_patch_or_localized_lr_difference_evidence"),
    ]

    deprioritize = ["larger_backbone"]
    if not is_mono:
        # Stereo-only failed mechanisms — irrelevant (not failures) for mono input.
        deprioritize += ["two_independent_backbone_stereo", "global_feature_difference_only"]
    deprioritize += ["threshold_only_fix", "full_freeze_only_when_prob_gap_flat"]

    if is_mono:
        augmentation_safety = (
            "Use AOI-safe augmentation: avoid heavy color jitter, random erasing over "
            "defects, and random perspective. Prefer mixup/CutMix and test-time "
            "augmentation for regularization on the small single-image dataset."
        )
    else:
        augmentation_safety = (
            "Sample geometric parameters once and apply identically to L/R; avoid "
            "heavy color jitter, random erasing over defects, random perspective, "
            "and L/R-desynchronizing affine/rotation/crop."
        )

    return {
        "train_sample_count": 287,
        "input_modality": input_modality,
        "prefer_order": prefer_order,
        "deprioritize": deprioritize,
        "known_failed_fingerprints": [
            [target, mechanism]
            for target, mechanism in sorted(KNOWN_FAILED_STRATEGY_FINGERPRINTS)
        ],
        "primary_signals": ["val_auc", "prob_gap"],
        "confirmation_signal": "test best recall@overkill<=0.08 over configured seeds",
        "augmentation_safety": augmentation_safety,
        "reporting_note": "Calibration/threshold curves are reporting, not the only fix.",
    }


def classify_failure_mode(
    baseline_metrics: dict,
    calibration_stats: Optional[dict] = None,
    error_analysis: Optional[dict] = None,
    threshold_curve: Optional[list] = None,
    input_modality: str = "stereo",
) -> dict:
    """
    Classify the dominant failure mode using the overkill decision tree (P2B).
    Returns dict with failure_mode, confidence, evidence, recommended_target, recommended_action.

    ``input_modality`` defaults to "stereo" (preserves all existing behaviour). When
    "mono", stereo-specific recommendations (9-channel L/R/diff, SSIM, stereo difference
    features) are replaced with single-image strategies (spatial attention, mixup/CutMix, TTA).
    """
    is_mono = input_modality == "mono"
    # Mono replacement for the stereo g_ng_overlap recommendation.
    mono_overlap_target = "feature_representation"
    mono_overlap_action = (
        "Improve single-image feature separability: add spatial attention on the "
        "backbone features, apply mixup/CutMix augmentation, and use a test-time "
        "augmentation (TTA) ensemble for more confident G/NG separation."
    )
    overkill = float(baseline_metrics.get("overkill_rate", 1.0))
    miss_rate = float(baseline_metrics.get("miss_rate", 1.0))
    accuracy = float(baseline_metrics.get("accuracy", 0.0))
    prob_gap = float(baseline_metrics.get("prob_gap", 0.0) or 0.0)

    # Extract G_prob_mean from calibration_stats or error_analysis
    g_prob_mean = None
    ng_prob_mean = None
    if calibration_stats:
        g_prob_mean = calibration_stats.get("G_prob_mean")
        ng_prob_mean = calibration_stats.get("NG_prob_mean")
    elif error_analysis and error_analysis.get("probability_summary"):
        ps = error_analysis["probability_summary"]
        if ps.get("G", {}).get("mean") is not None:
            g_prob_mean = ps["G"]["mean"]
        if ps.get("NG", {}).get("mean") is not None:
            ng_prob_mean = ps["NG"]["mean"]

    if prob_gap <= config.PROBE_PROBABILITY_GAP_MIN:
        return {
            "failure_mode": "full_freeze_underfit",
            "confidence": "high",
            "evidence": (
                f"prob_gap={prob_gap:.3f} is at/below the separability floor "
                f"({config.PROBE_PROBABILITY_GAP_MIN:.3f}); mg9 full-freeze runs "
                "showed this pattern as flat predictions/underfit."
            ),
            "recommended_target": "model_architecture",
            "recommended_action": (
                "Use partial-unfreeze adaptation: freeze early ResNet18 backbone "
                "layers, unfreeze only layer4/final block, keep a small dropout "
                "head, AdamW, and nonzero weight_decay. Prefer local/ROI/patch "
                "evidence next; do not retry full-freeze-only or threshold-only fixes."
            ),
        }

    # --- P0 check first: miss rate is the highest-priority metric (spec §8) ---
    # When miss_rate is failing, FP-reduction strategies are contraindicated — they
    # shift the threshold toward conservatism and worsen FN / defect escape.
    # Overkill (P2) is only addressed once P0 is satisfied.
    if miss_rate > 0.03:
        if overkill > 0.40:
            # Both are failing; P0 leads. Do NOT apply FP penalty here — it will
            # worsen miss. Target recall / separability first.
            return {
                "failure_mode": "low_capacity_miss",
                "confidence": "high",
                "evidence": (
                    f"miss_rate={miss_rate:.3f} violates P0 threshold (>0.03); "
                    f"overkill={overkill:.3f} also high but P0 takes priority. "
                    "FP-reduction strategies are contraindicated while miss_rate > 0.03."
                ),
                "recommended_target": "model_architecture",
                "recommended_action": (
                    "Miss rate is above the P0 threshold — do NOT add FP penalty loss "
                    "(it increases FN / defect escape). Target recall: increase model "
                    "capacity, add defect-region attention, or improve single-image "
                    "feature separability so NG samples are more confidently detected."
                    if is_mono else
                    "Miss rate is above the P0 threshold — do NOT add FP penalty loss "
                    "(it increases FN / defect escape). Target recall: increase model "
                    "capacity, add defect-region attention, or improve stereo feature "
                    "separability so NG samples are more confidently detected."
                ),
            }
        else:
            # Only miss_rate failing; overkill is acceptable.
            return {
                "failure_mode": "low_capacity_miss",
                "confidence": "medium",
                "evidence": f"miss_rate={miss_rate:.3f} (P0 violation), overkill={overkill:.3f} (within budget)",
                "recommended_target": "model_architecture",
                "recommended_action": "Increase model capacity or add attention to defect regions.",
            }

    # --- Overkill decision tree (P2): only reached when miss_rate <= 0.03 ---
    if overkill > 0.40 and threshold_curve:
        curve_summary = summarize_threshold_curve(threshold_curve)
        threshold_can_escape_overkill = any(
            _threshold_point_overkill(point) < 0.20 for point in threshold_curve
        )
        if not threshold_can_escape_overkill:
            return {
                "failure_mode": "g_ng_overlap",
                "confidence": "high",
                "evidence": (
                    f"overkill={overkill:.3f}; threshold_curve has "
                    f"{curve_summary['points']} points but no point reaches overkill<0.20"
                ),
                "recommended_target": mono_overlap_target if is_mono else "stereo_fusion",
                "recommended_action": (
                    "Thresholding cannot solve this failure. Force representation work: "
                    + mono_overlap_action
                    if is_mono else
                    "Thresholding cannot solve this failure. Force representation/stereo "
                    "separability work: 9-channel L/R/diff input, stronger difference features, "
                    "or SSIM/attention features before further threshold tuning."
                ),
            }

    if overkill > 0.40:
        if g_prob_mean is not None and g_prob_mean > 0.50:
            return {
                "failure_mode": "threshold_collapse",
                "confidence": "high",
                "evidence": f"overkill={overkill:.3f}, G_prob_mean={g_prob_mean:.3f} (>0.50)",
                "recommended_target": "weighted_loss",
                "recommended_action": (
                    "Add constraint-aware asymmetric loss with dynamic FP penalty: "
                    "fp_weight = 1.0 + 5.0 * max(0, best_overkill_rate - 0.08). "
                    "Use constrained threshold selection."
                ),
            }
        elif g_prob_mean is not None and 0.20 <= g_prob_mean <= 0.50:
            return {
                "failure_mode": "g_ng_overlap",
                "confidence": "high",
                "evidence": f"overkill={overkill:.3f}, G_prob_mean={g_prob_mean:.3f} (0.20-0.50)",
                "recommended_target": mono_overlap_target if is_mono else "stereo_fusion",
                "recommended_action": (
                    mono_overlap_action
                    if is_mono else
                    "Improve L/R stereo difference features: emphasize abs(L-R) channel, "
                    "add SSIM map, use difference-attention module for better G/NG separability."
                ),
            }
        elif g_prob_mean is not None and g_prob_mean < 0.20:
            return {
                "failure_mode": "preprocessing_lot_shift",
                "confidence": "medium",
                "evidence": f"overkill={overkill:.3f}, G_prob_mean={g_prob_mean:.3f} (<0.20)",
                "recommended_target": "preprocessing_normalization",
                "recommended_action": (
                    "Add per-lot normalization (compute mean/std per lot folder from train only). "
                    "Apply CLAHE contrast equalization to each image."
                    if is_mono else
                    "Add per-lot normalization (compute mean/std per lot folder from train only). "
                    "Apply CLAHE contrast equalization to L and R before diff computation."
                ),
            }
        else:
            return {
                "failure_mode": "g_ng_overlap",
                "confidence": "low",
                "evidence": f"overkill={overkill:.3f}, no calibration stats available",
                "recommended_target": mono_overlap_target if is_mono else "stereo_fusion",
                "recommended_action": (
                    "Ensure CALIBRATION_STATS output is present. "
                    "Add spatial attention on backbone features, mixup/CutMix, and "
                    "probability calibration."
                    if is_mono else
                    "Ensure CALIBRATION_STATS output is present. "
                    "Improve L/R diff features and add probability calibration."
                ),
            }

    # Near acceptance
    dist = acceptance_distance(baseline_metrics)
    if dist < 0.5:
        return {
            "failure_mode": "near_acceptance",
            "confidence": "high",
            "evidence": f"acceptance_distance={dist:.3f} (<0.5)",
            "recommended_target": "calibration",
            "recommended_action": "Add temperature scaling calibration + fine-tune threshold.",
        }

    # Default
    return {
        "failure_mode": "g_ng_overlap",
        "confidence": "low",
        "evidence": f"overkill={overkill:.3f}, miss_rate={miss_rate:.3f}, accuracy={accuracy:.3f}",
        "recommended_target": mono_overlap_target if is_mono else "stereo_fusion",
        "recommended_action": (
            "Improve feature extraction with spatial attention and mixup/CutMix augmentation."
            if is_mono else
            "Improve feature extraction with stereo difference emphasis."
        ),
    }


def compute_ablation_deltas(
    ablation_results: list,
    baseline_metrics: dict,
) -> list:
    """Compute deltas for each ablation variant vs baseline. Returns sorted list."""
    baseline = metrics_view(baseline_metrics)
    baseline_dist = acceptance_distance(baseline_metrics)
    ranked = []

    for result in ablation_results:
        if not isinstance(result, dict):
            continue
        name = result.get("name", "unknown")
        status = result.get("status", "unknown")
        variant_metrics = result.get("metrics")

        if status != "success" or not variant_metrics:
            ranked.append({
                "name": name,
                "status": status,
                "variant_index": result.get("variant_index"),
                "impact_score": 0.0,
                "deltas": None,
                "note": "failed/skipped — tightly coupled component",
            })
            continue

        vm = metrics_view(variant_metrics)
        is_probe = name.endswith("_probe")
        deltas = {
            "delta_miss_rate": round(vm["miss_rate"] - baseline["miss_rate"], 4),
            "delta_ng_recall": round(baseline["ng_recall"] - vm["ng_recall"], 4),
            "delta_overkill": round(vm["overkill_rate"] - baseline["overkill_rate"], 4),
            "delta_accuracy": round(baseline["accuracy"] - vm["accuracy"], 4),
        }
        if is_probe:
            deltas["improvement_overkill"] = round(baseline["overkill_rate"] - vm["overkill_rate"], 4)
            deltas["improvement_accuracy"] = round(vm["accuracy"] - baseline["accuracy"], 4)

        variant_dist = acceptance_distance(vm)
        impact_score = round(abs(baseline_dist - variant_dist), 4)
        # "helpful" = removal of this component made things better (component was hurting).
        # "harmful" = removal made things worse (component was helping).
        # Signed direction lets the planner distinguish the two without ambiguity.
        if variant_dist < baseline_dist:
            impact_direction = "helpful"
        elif variant_dist > baseline_dist:
            impact_direction = "harmful"
        else:
            impact_direction = "neutral"

        ranked.append({
            "name": name, "status": status,
            "variant_index": result.get("variant_index"),
            "metrics": {k: round(v, 4) for k, v in vm.items()},
            "deltas": deltas,
            "impact_score": impact_score,
            "impact_direction": impact_direction,
            "acceptance_distance": round(variant_dist, 4),
            "is_probe": is_probe,
        })

    ranked.sort(key=lambda r: (
        0 if r.get("status") != "success" else 1,
        -r.get("impact_score", 0),
    ))
    return ranked


def generate_diagnosis_brief(
    ablation_results: list,
    baseline_metrics: dict,
    calibration_stats: Optional[dict] = None,
    error_analysis: Optional[dict] = None,
    threshold_curve: Optional[list] = None,
    input_modality: str = "stereo",
) -> dict:
    """Generate a complete structured diagnosis brief for the LLM.

    ``input_modality`` defaults to "stereo" (preserves all existing behaviour) and is
    forwarded to the failure classifier and small-data policy so mono runs receive
    single-image recommendations instead of stereo-specific ones.
    """
    ranked = compute_ablation_deltas(ablation_results, baseline_metrics)
    curve = threshold_curve or []
    failure = classify_failure_mode(
        baseline_metrics, calibration_stats, error_analysis, threshold_curve=curve,
        input_modality=input_modality,
    )

    best_probe = next(
        (r for r in ranked if r.get("is_probe") and r.get("deltas", {}).get("improvement_overkill", 0) > 0),
        None,
    )
    most_impactful_removal = next(
        (r for r in ranked if not r.get("is_probe") and r.get("status") == "success"),
        None,
    )

    return {
        "failure_classification": failure,
        "ablation_ranking": ranked,
        "best_probe": best_probe,
        "most_impactful_removal": most_impactful_removal,
        "baseline_metrics": {k: round(v, 4) for k, v in metrics_view(baseline_metrics).items()},
        "baseline_acceptance_distance": round(acceptance_distance(baseline_metrics), 4),
        "threshold_curve": curve,
        "threshold_curve_summary": summarize_threshold_curve(curve),
        "recommended_target": failure["recommended_target"],
        "recommended_action": failure["recommended_action"],
        "small_data_strategy_policy": small_data_strategy_policy(input_modality),
    }


def summarize_threshold_curve(threshold_curve: list) -> dict:
    """Return compact trade-off landmarks for planner/diagnosis prompts."""
    if not threshold_curve:
        return {"points": 0, "best_acceptance_distance": None, "best_threshold": None}

    def _float_or_default(value, default):
        if value is None:
            return default
        return float(value)

    scored = []
    for point in threshold_curve:
        try:
            metrics = {
                "accuracy": _float_or_default(point.get("accuracy"), 0.0),
                "ng_recall": _float_or_default(point.get("recall", point.get("ng_recall")), 0.0),
                "miss_rate": _float_or_default(point.get("miss_rate"), 1.0),
                "overkill_rate": _float_or_default(point.get("overkill", point.get("overkill_rate")), 1.0),
                "f1": _float_or_default(point.get("f1"), 0.0),
            }
        except (TypeError, ValueError):
            continue
        scored.append((acceptance_distance(metrics), point))

    if not scored:
        return {"points": len(threshold_curve), "best_acceptance_distance": None, "best_threshold": None}

    best_distance, best_point = min(scored, key=lambda item: item[0])
    return {
        "points": len(threshold_curve),
        "best_acceptance_distance": round(best_distance, 4),
        "best_threshold": best_point.get("t", best_point.get("threshold")),
        "best_point": best_point,
    }


def _threshold_point_overkill(point: dict) -> float:
    """Return an overkill value from either threshold-curve field spelling."""
    try:
        return float(point.get("overkill", point.get("overkill_rate", 1.0)))
    except (TypeError, ValueError):
        return 1.0


# ---------------------------------------------------------------------------
# Parsing helpers for new training signals (P1)
# ---------------------------------------------------------------------------

def parse_calibration_stats(stdout: str) -> Optional[dict]:
    """Parse CALIBRATION_STATS: {...} from stdout."""
    result = _extract_json_block(stdout, "CALIBRATION_STATS")
    return result if isinstance(result, dict) else None


def parse_threshold_curve(stdout: str) -> Optional[list]:
    """Parse THRESHOLD_CURVE: [...] from stdout."""
    match = re.search(r"THRESHOLD_CURVE:\s*(\[.*?\])", stdout, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def parse_epoch_logs(stdout: str) -> list:
    """Parse all EPOCH_LOG: {...} lines from stdout."""
    results = []
    for match in re.finditer(r"EPOCH_LOG:\s*(\{.*?\})", stdout):
        try:
            results.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return results


def detect_early_collapse(epoch_logs: list) -> Optional[dict]:
    """Detect if model collapsed to predict-all-NG early in training."""
    if len(epoch_logs) < 3:
        return None
    early = epoch_logs[:3]
    high_overkill_early = all(
        e.get("val_overkill", 0) > 0.50 for e in early
    )
    if high_overkill_early:
        return {
            "detected": True,
            "pattern": "val_overkill > 0.50 in first 3 epochs",
            "recommendation": "Model is collapsing to predict-all-NG. Change loss function or add FP penalty.",
            "epoch_details": early,
        }
    return {"detected": False}
