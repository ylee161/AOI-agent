"""Static checks for small-data-safe AOI strategy constraints.

These helpers are intentionally conservative. They are used by the prompt-level
validator to flag risky candidates before an expensive training run, especially
changes that add capacity on the 287-sample grouped train split without freeze,
dropout, or weight decay.
"""

from __future__ import annotations

import ast
from typing import Any


KNOWN_FAILED_STRATEGY_FINGERPRINTS = {
    ("threshold_sweep", "threshold_acceptance_distance"),
    ("calibration", "threshold_only"),
    ("stereo_fusion", "global_feature_difference_only"),
    ("stereo_fusion", "two_independent_backbone_stereo"),
    ("model_architecture", "larger_backbone"),
    # MG11: any geometric augmentation on the ROI crop collapses prob_gap
    # (v3_roi_centered_aug: val prob_gap 0.058 < 0.10 over-aug signature).
    ("augmentation", "roi_geometric_augmentation"),
    # MG15/MG16: the TRAINING-FREE / UNSUPERVISED per-board anomaly detector route
    # (isolation forest, one-class SVM, SVDD, autoencoder reconstruction-error, etc.)
    # — the 0.78 AUC was a board-pooling eval leak; clean held-out AUC 0.50, 83%
    # overkill. This bans only the label-free anomaly/OOD framing. It does NOT ban
    # SUPERVISED anomaly-detection models trained with G/NG labels (e.g. PatchCore,
    # EfficientAD): those learn from the labels and are a distinct mechanism, so they
    # carry their own mechanism_class (e.g. "patchcore", "efficientad") and remain
    # allowed for empirical evaluation.
    ("model_architecture", "unsupervised_anomaly_detector"),
    # MG14: local-patch / multiple-instance-learning route — AUC 0.45, 98% overkill.
    ("model_architecture", "local_patch_mil"),
}

_LARGE_BACKBONES = {
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "convnext",
    "vit",
    "swin",
}

_UNSAFE_AUGMENTATIONS = {
    "ColorJitter",
    "RandomErasing",
    "RandomPerspective",
}

_GEOMETRIC_AUGMENTATIONS = {
    "RandomAffine",
    "RandomRotation",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomResizedCrop",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _literal_int(node.left)
        right = _literal_int(node.right)
        if left is not None and right is not None:
            return left * right
    return None


def _string_constants(tree: ast.AST) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _has_requires_grad_false(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr == "requires_grad":
                value = node.value
                if isinstance(value, ast.Constant) and value.value is False:
                    return True
    return False


def _has_partial_unfreeze_last_block(tree: ast.AST, strings: list[str]) -> bool:
    """Detect the small-data-safe pattern: freeze broadly, then unfreeze layer4/final block."""
    text = "\n".join(strings).lower()
    if any(token in text for token in ("layer4", "final resnet block", "last resnet block")):
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr == "requires_grad":
                    value = node.value
                    if isinstance(value, ast.Constant) and value.value is True:
                        return True
    return False


def _linear_param_count(call: ast.Call) -> int | None:
    name = _call_name(call.func).split(".")[-1]
    if name != "Linear" or len(call.args) < 2:
        return None
    in_features = _literal_int(call.args[0])
    out_features = _literal_int(call.args[1])
    if in_features is None or out_features is None:
        return None
    return in_features * out_features


def _has_independent_stereo_backbones(tree: ast.AST) -> bool:
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = _call_name(node.value.func).lower()
        if not any(backbone in call for backbone in ("resnet", "efficientnet", "convnext", "vit", "swin")):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                assignments[target.attr.lower()] = call
    left_keys = {k for k in assignments if k in {"left", "left_backbone", "left_encoder", "encoder_l", "backbone_l"}}
    right_keys = {k for k in assignments if k in {"right", "right_backbone", "right_encoder", "encoder_r", "backbone_r"}}
    return bool(left_keys and right_keys)


def _has_global_feature_difference_only(tree: ast.AST, strings: list[str]) -> bool:
    lowered = " ".join(strings).lower()
    declares_feature_diff = "feature_diff_candidate" in lowered or "feature-level" in lowered
    has_concat_diff = False
    has_local_evidence = any(token in lowered for token in ("local", "patch", "roi", "localized"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func).endswith("cat"):
            rendered = ast.dump(node).lower()
            if "f_l" in rendered and "f_r" in rendered and "diff" in rendered:
                has_concat_diff = True
    return (declares_feature_diff or has_concat_diff) and not has_local_evidence


def _metric_reporting_present(tree: ast.AST, strings: list[str]) -> bool:
    text = "\n".join(strings)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return (
        ("roc_auc" in text or "roc_auc" in names)
        and ("prob_gap" in text or "prob_gap" in names)
        and "THRESHOLD_CURVE:" in text
    )


def _has_degenerate_prediction_guard(strings: list[str], tree: ast.AST) -> bool:
    text = "\n".join(strings)
    if "DEGENERATE_PREDICTION_WARNING" in text or "DEGENERATE_THRESHOLD_WARNING" in text:
        return True
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return bool({"score_range", "unique_scores", "score_std"} & names)


def _augmentation_report(tree: ast.AST, input_modality: str = "stereo") -> tuple[bool, bool]:
    call_names = [_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    short_names = {name.split(".")[-1] for name in call_names}
    has_unsafe = bool(short_names & _UNSAFE_AUGMENTATIONS)
    has_geometric = bool(short_names & _GEOMETRIC_AUGMENTATIONS)
    has_get_params = any(name.endswith(".get_params") for name in call_names)
    dumps = [ast.dump(node).lower() for node in ast.walk(tree) if isinstance(node, ast.Call)]
    paired_l = any("img_l" in dump and "affine" in dump for dump in dumps)
    paired_r = any("img_r" in dump and "affine" in dump for dump in dumps)
    paired = has_get_params and paired_l and paired_r
    # Mono datasets have no L/R pair to desynchronise, so geometric augmentation
    # without L/R pairing is not inherently unsafe — only the genuinely unsafe
    # transforms are flagged.
    unsafe = has_unsafe if input_modality == "mono" else has_unsafe or (has_geometric and not paired)
    return unsafe, paired


def validate_small_data_strategy_source(script: str, input_modality: str = "stereo") -> dict[str, Any]:
    """Return a static report for small-data AOI strategy safety."""
    try:
        tree = ast.parse(script)
    except SyntaxError as exc:
        return {
            "ok": False,
            "syntax_error": str(exc),
            "reasons": ["syntax_error"],
        }

    strings = _string_constants(tree)
    call_names = [_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    short_names = {name.split(".")[-1] for name in call_names}

    uses_legacy_split = any(
        "checkpoints/data_split.json" in s and "data_split_grouped.json" not in s
        for s in strings
    )
    has_freeze = _has_requires_grad_false(tree)
    has_partial_unfreeze = _has_partial_unfreeze_last_block(tree, strings)
    has_dropout = bool(short_names & {"Dropout", "Dropout2d", "AlphaDropout"})
    has_weight_decay = any(
        isinstance(node, ast.keyword)
        and node.arg == "weight_decay"
        and not (isinstance(node.value, ast.Constant) and float(node.value.value or 0.0) == 0.0)
        for node in ast.walk(tree)
    )

    linear_params = [
        count
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for count in [_linear_param_count(node)]
        if count is not None
    ]
    has_large_head = any(count >= 300_000 for count in linear_params)
    has_large_backbone = any(name.split(".")[-1].lower() in _LARGE_BACKBONES for name in call_names)
    is_stereo = input_modality == "stereo"
    # Two-independent-backbone stereo is a stereo-only failure fingerprint; it
    # cannot occur for mono inputs, so skip the check entirely when mono.
    independent_backbones = _has_independent_stereo_backbones(tree) if is_stereo else False
    large_capacity = has_large_head or has_large_backbone or independent_backbones
    has_regularization = has_freeze or has_dropout or has_weight_decay

    unsafe_augmentation, paired_aug = _augmentation_report(tree, input_modality)
    missing_metric_reporting = not _metric_reporting_present(tree, strings)
    missing_degenerate_guard = not _has_degenerate_prediction_guard(strings, tree)
    # Global feature-difference-only is likewise a stereo-only fingerprint.
    global_feature_difference_only = (
        _has_global_feature_difference_only(tree, strings) if is_stereo else False
    )
    known_failed = (
        global_feature_difference_only
        or has_large_backbone
        or independent_backbones
    )

    reasons = []
    if uses_legacy_split:
        reasons.append("uses_legacy_split")
    if large_capacity and not has_regularization:
        reasons.append("large_capacity_without_regularization")
    if unsafe_augmentation:
        reasons.append("unsafe_augmentation")
    if missing_metric_reporting:
        reasons.append("missing_metric_reporting")
    if missing_degenerate_guard:
        reasons.append("missing_degenerate_guard")
    if known_failed:
        reasons.append("known_failed_fingerprint")

    warnings = []
    full_freeze_without_partial_unfreeze = has_freeze and not has_partial_unfreeze
    if full_freeze_without_partial_unfreeze:
        warnings.append("full_freeze_without_partial_unfreeze")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "uses_legacy_split": uses_legacy_split,
        "has_freeze": has_freeze,
        "has_partial_unfreeze": has_partial_unfreeze,
        "full_freeze_without_partial_unfreeze": full_freeze_without_partial_unfreeze,
        "has_dropout": has_dropout,
        "has_weight_decay": has_weight_decay,
        "large_capacity_without_regularization": large_capacity and not has_regularization,
        "large_added_capacity": large_capacity,
        "has_large_head": has_large_head,
        "has_large_backbone": has_large_backbone,
        "independent_backbones": independent_backbones,
        "unsafe_augmentation": unsafe_augmentation,
        "has_aoi_safe_paired_augmentation": paired_aug,
        "missing_metric_reporting": missing_metric_reporting,
        "missing_degenerate_guard": missing_degenerate_guard,
        "known_failed_fingerprint": known_failed,
        "known_failed_strategy_fingerprints": sorted(KNOWN_FAILED_STRATEGY_FINGERPRINTS),
    }
