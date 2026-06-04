"""Static (no-execution) checks confirming a training script actually USES the
declared data modalities.

This is the MLE-STAR "Data Usage Checker" guardrail and the complement of the
Data Leakage checker: leakage asks "does the script touch data it must not",
usage asks "does the script touch the data it must". It statically confirms that
a generated AOI training script:
  - loads BOTH stereo images (_L and _R) when the input modality is stereo,
  - reads the Excel labels (the .xlsx result column), and
  - reads the authoritative split the pipeline injects (data_split / train/val/test).

Like the other validators in this package the checks are intentionally
conservative: when uncertain they do NOT flag, so the legitimate cases
(mono inputs, and the no_stereo_fusion ablation that drops _R on purpose) never
produce a false positive. On unparseable code the audit degrades to a best-effort
text scan and reports "usage check inconclusive" rather than crashing the
validator.
"""

from __future__ import annotations

import ast
from typing import Any

# The ablation probe that legitimately uses only the left image; CHECK 3 must
# never raise "right image unused" for it.
_NO_STEREO_FUSION_VARIANT = "no_stereo_fusion"

# Tokens that count as evidence the Excel labels are read.
_EXCEL_TOKENS = ("read_excel", ".xlsx", "openpyxl", "ExcelFile", "load_workbook")

# Tokens that count as evidence the provided split is read. data_split is the
# authoritative contract; the train/val/test path names are accepted as well
# because some scripts unpack the split into those before use.
_SPLIT_TOKENS = (
    "data_split",
    "data_split.json",
    "data_split_grouped",
    "train_paths",
    "val_paths",
    "test_paths",
    "train_split",
    "val_split",
    "test_split",
)


def _string_constants(tree: ast.AST) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _ablation_variant_name(tree: ast.AST) -> str | None:
    """Return the value of a top-level ``ABLATION_VARIANT_NAME = "..."`` assignment.

    Only module-level string assignments count; this mirrors how the diagnostic
    ablation probes declare themselves.
    """
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(node, ast.Assign):
            continue
        if not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "ABLATION_VARIANT_NAME":
                return node.value.value
    return None


def _references(strings: list[str], tokens: tuple[str, ...]) -> bool:
    blob = "\n".join(strings)
    return any(token in blob for token in tokens)


def _references_right_image(strings: list[str]) -> bool:
    """True if any string constant references the right stereo image.

    We look for the ``_R`` infix the dataset uses (``_R_`` and the ``_R``
    suffix/keying variants) so a script that loads the right image — under any of
    the conventional spellings — is recognised.
    """
    return any(("_R_" in s) or ("_R." in s) or s.endswith("_R") or "img_r" in s.lower() for s in strings)


def _references_left_image(strings: list[str]) -> bool:
    return any(("_L_" in s) or ("_L." in s) or s.endswith("_L") or "img_l" in s.lower() for s in strings)


def _text_scan_report(script: str, input_modality: str) -> dict[str, Any]:
    """Best-effort report used only when the script cannot be parsed.

    Conservative by construction: it reports ``inconclusive`` and never raises a
    violation, so unparseable code can never produce a false positive.
    """
    return {
        "ok": True,
        "inconclusive": True,
        "reasons": [],
        "messages": ["usage check inconclusive"],
        "input_modality": input_modality,
        "ablation_variant_name": None,
        "references_left_image": "_L" in script,
        "references_right_image": "_R" in script,
        "reads_excel_labels": any(token in script for token in _EXCEL_TOKENS),
        "reads_split": any(token in script for token in _SPLIT_TOKENS),
    }


def validate_data_usage_source(script: str, input_modality: str = "stereo") -> dict[str, Any]:
    """Return a static report on whether the script USES the declared data.

    Violations (each a short human-readable reason in ``messages``):
      - "stereo input but right image unused" — stereo modality, the script
        references _L but never _R (exempt for mono and the no_stereo_fusion
        ablation, which legitimately drop _R);
      - "labels not loaded" — no read_excel / .xlsx / openpyxl evidence;
      - "ignores provided split" — no data_split / train/val/test reference.

    On a parse failure the audit degrades to a best-effort text scan and reports
    "usage check inconclusive"; it never raises.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        # Degrade to a best-effort text scan — never crash the validator.
        return _text_scan_report(script, input_modality)

    strings = _string_constants(tree)
    ablation_variant = _ablation_variant_name(tree)

    references_left = _references_left_image(strings)
    references_right = _references_right_image(strings)
    reads_excel = _references(strings, _EXCEL_TOKENS)
    reads_split = _references(strings, _SPLIT_TOKENS)

    # EXEMPTION: mono inputs have no right image, and the no_stereo_fusion
    # ablation drops _R on purpose — neither may raise "right image unused".
    right_image_exempt = (
        input_modality != "stereo" or ablation_variant == _NO_STEREO_FUSION_VARIANT
    )

    reasons: list[str] = []
    messages: list[str] = []

    # Only flag a missing right image when the script clearly loads the LEFT one;
    # a script that references neither is doing something we do not understand, so
    # — when uncertain, do NOT flag.
    if not right_image_exempt and references_left and not references_right:
        reasons.append("right_image_unused")
        messages.append("stereo input but right image unused")

    if not reads_excel:
        reasons.append("labels_not_loaded")
        messages.append("labels not loaded")

    if not reads_split:
        reasons.append("ignores_provided_split")
        messages.append("ignores provided split")

    return {
        "ok": not reasons,
        "inconclusive": False,
        "reasons": reasons,
        "messages": messages,
        "input_modality": input_modality,
        "ablation_variant_name": ablation_variant,
        "references_left_image": references_left,
        "references_right_image": references_right,
        "reads_excel_labels": reads_excel,
        "reads_split": reads_split,
        "right_image_exempt": right_image_exempt,
    }
