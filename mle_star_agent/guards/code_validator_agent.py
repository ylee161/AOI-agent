import hashlib

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools import agent_tool

from mle_star_agent import config
from mle_star_agent.shared import code_runner
from mle_star_agent.shared.callbacks import count_tokens_callback, rate_limit_retry_callback
from mle_star_agent.shared.difference_feature_validator import (
    validate_difference_feature_source,
)
from mle_star_agent.shared.small_data_strategy_validator import (
    validate_small_data_strategy_source,
)

# ---------------------------------------------------------------------------
# Dry-run environment injected during validation.
# Scripts must read these vars and cap epochs/samples accordingly so the
# validator completes in seconds rather than running full training.
# ---------------------------------------------------------------------------
_DRY_RUN_ENV = {
    "DRY_RUN": "1",
    "DRY_RUN_EPOCHS": "1",
    "DRY_RUN_SAMPLES": "10",
    "AOI_RANDOM_SEED": "42",
    "PYTHONHASHSEED": "42",
    "SEED": "42",
}

_VALIDATOR_TIMEOUT = 120  # seconds — enough for 1 epoch on 10 samples, even on CPU
_REQUIRED_FULL_RUN_MARKERS = (
    "PROBE_METRICS:",
    "CALIBRATION_STATS:",
    "THRESHOLD_CURVE:",
    "PREDICTIONS:",
)


# ---------------------------------------------------------------------------
# FunctionTools
# ---------------------------------------------------------------------------

def run_script_fn(script: str) -> dict:
    """Execute a Python script in dry-run mode and return stdout, stderr, and exit code.

    Dry-run env vars injected: DRY_RUN=1, DRY_RUN_EPOCHS=1, DRY_RUN_SAMPLES=10.
    Scripts must honour these to cap training to a few seconds.
    """
    result = code_runner.run_script(script, timeout=_VALIDATOR_TIMEOUT, env=_DRY_RUN_ENV)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-5000:],
        "stderr": result.stderr[-5000:],
        "timed_out": result.timed_out,
        "duration_ms": round(result.duration_ms, 1),
    }


def missing_required_full_run_markers(script: str) -> list[str]:
    """Return full-run diagnostic markers missing from generated script text."""
    return [marker for marker in _REQUIRED_FULL_RUN_MARKERS if marker not in script]


def static_contract_check_fn(script: str) -> dict:
    missing = missing_required_full_run_markers(script)
    return {
        "ok": not missing,
        "missing_markers": missing,
    }


def static_fp_penalty_check_fn(script: str) -> dict:
    """Detect whether a script implements the required dynamic false-positive penalty."""
    compact = script.replace(" ", "")
    has_dynamic_fp_weight = (
        "fp_weight" in script
        and "max(0" in compact
        and "0.08" in script
    )
    has_fp_loss_term = any(token in script for token in (
        "false_positive_loss",
        "fp_loss",
        "g_false_positive",
        "false_positive_penalty",
    ))
    return {
        "ok": has_dynamic_fp_weight and has_fp_loss_term,
        "has_dynamic_fp_weight": has_dynamic_fp_weight,
        "has_fp_loss_term": has_fp_loss_term,
    }


def static_degenerate_threshold_guard_check_fn(script: str) -> dict:
    """Detect whether a script reports degenerate low-threshold/high-overkill results."""
    compact = script.replace(" ", "")
    has_threshold_condition = any(token in compact for token in (
        "threshold<=0.15",
        "best_threshold<=0.15",
        "selected_threshold<=0.15",
    ))
    has_overkill_condition = any(token in compact for token in (
        "overkill>0.50",
        "overkill_rate>0.50",
        "val_overkill>0.50",
    ))
    has_warning = "DEGENERATE_THRESHOLD_WARNING" in script
    return {
        "ok": has_threshold_condition and has_overkill_condition and has_warning,
        "has_threshold_condition": has_threshold_condition,
        "has_overkill_condition": has_overkill_condition,
        "has_warning": has_warning,
    }


# Markers a candidate uses to opt into the feature-level Siamese difference family.
# The marker only ROUTES to the structural check below; the check itself is AST-based
# on the actual model code, not on the presence of these strings.
_FEATURE_DIFF_MARKERS = (
    "FEATURE_DIFF_CANDIDATE",
    "feature_diff",
    "feature-level difference",
    "siamese_difference",
    "siamese difference",
)


def declares_feature_diff_candidate(script: str) -> bool:
    """True if the script opts into the feature-level Siamese difference family."""
    lowered = script.lower()
    return any(marker.lower() in lowered for marker in _FEATURE_DIFF_MARKERS)


def static_difference_feature_check_fn(script: str) -> dict:
    """AST check that a feature-level shared-weight Siamese difference is actually used.

    This is NOT the pixel-level 9-channel abs(img_l - img_r) input. It confirms, via
    AST dataflow over the model's forward method, that:
      - a single shared encoder (`self.<attr>`) processes left -> f_L and right -> f_R,
      - abs(f_L - f_R) is computed on the two ENCODER OUTPUTS (feature level),
      - the classifier head receives concat [f_L, f_R, |f_L - f_R|].

    `applies` is True only when the script declared itself part of this family; when
    False the check is informational and does not gate validation.
    """
    report = validate_difference_feature_source(script)
    report["applies"] = declares_feature_diff_candidate(script)
    return report


def static_small_data_strategy_check_fn(tool_context, script: str) -> dict:
    """AST/static check for small-data-safe strategy constraints."""
    input_modality = tool_context.state.get("input_modality", "stereo")
    return validate_small_data_strategy_source(script, input_modality=input_modality)


def append_failed_script_fn(tool_context, script_name: str, error: str, attempts: int) -> str:
    """Record a script that exhausted all debug retry attempts into state['failed_scripts']."""
    failed = list(tool_context.state.get("failed_scripts", []) or [])
    failed.append({"script_name": script_name, "error": error, "attempts": attempts})
    tool_context.state["failed_scripts"] = failed
    return f"Recorded failed script '{script_name}' after {attempts} attempt(s)."


run_script_tool = FunctionTool(func=run_script_fn)
static_contract_check_tool = FunctionTool(func=static_contract_check_fn)
static_fp_penalty_check_tool = FunctionTool(func=static_fp_penalty_check_fn)
static_degenerate_threshold_guard_check_tool = FunctionTool(func=static_degenerate_threshold_guard_check_fn)
static_difference_feature_check_tool = FunctionTool(func=static_difference_feature_check_fn)
static_small_data_strategy_check_tool = FunctionTool(func=static_small_data_strategy_check_fn)
_append_failed_script_tool = FunctionTool(func=append_failed_script_fn)

# ---------------------------------------------------------------------------
# Validator agent
# ---------------------------------------------------------------------------

_INSTRUCTION = f"""You are a strict code validator for AOI (Automated Optical Inspection) ML pipeline scripts.

You receive a Python script in the user message. Perform the following checks IN ORDER.

---
## CHECK 1 — Data Leakage (Static Analysis, no execution)

Read the script and detect ANY of these leakage patterns:
- Scaler / normalizer fit on val or test split (must be fit on train only, then transform val/test)
- Test labels used during training or threshold tuning
- Threshold optimised on the test set instead of the validation set
- Augmentation or feature extraction applied to test data using train statistics

If leakage is found: rewrite the script to eliminate it before proceeding.

---
## CHECK 2 — Stereo Image Usage (Static Analysis, no execution)

Read the (possibly rewritten) script and verify:
1. Both `_L` and `_R` stereo images are loaded and used as model input for every sample
2. Labels are loaded from either `data_split.json` (the preferred data contract) OR directly from an Excel file — either approach is valid

If stereo loading is absent or incorrect: rewrite the script to fix it before proceeding.
If labels are not loaded at all: rewrite the script to load them from `data_split.json`.

---
## CHECK 2B — Training Length and Threshold Policy (Static Analysis, no execution)

Read the script and verify:
1. Full runs train for at least {config.MIN_EPOCHS} epochs with early stopping patience
   {config.EARLY_STOPPING_PATIENCE}. Any value below {config.MIN_EPOCHS} in the non-dry-run
   branch is invalid and MUST be rewritten (e.g. `epochs = DRY_RUN_EPOCHS if DRY_RUN else 5`
   or `epochs = DRY_RUN_EPOCHS if DRY_RUN else 15` are both invalid).
2. DRY_RUN mode may still use `DRY_RUN_EPOCHS`; only the non-dry-run branch is constrained.
3. If the script identifies itself as `ABLATION_VARIANT_NAME = "threshold_acceptance_distance"`,
   acceptance-distance minimization is allowed for this diagnostic probe only. It must still
   choose thresholds on the validation set, keep miss_rate as the highest-priority gap, and
   still print the standard METRICS keys including roc_auc and prob_gap.
4. For all other scripts, threshold selection on validation must follow strict two-stage priority:
   Stage 0 — filter out thresholds with FP > 2 before selecting a threshold. FP <= 2
   is mandatory for the current 30-G validation/test scale. If no threshold survives,
   choose the threshold with the minimum FP count, then lowest miss_rate, and report
   the result as below-target rather than accepting high overkill.
   Stage 1 — find the threshold that minimises miss_rate (target: <= {config.MISS_RATE_RELAXED_MAX}).
   Stage 2 — among ALL thresholds that achieve that minimum miss_rate, pick the one with the
   lowest overkill_rate (target: <= {config.OVERKILL_RELAXED_MAX}).
   Do NOT use acceptance-distance averaging or a blended score as the primary objective —
   miss_rate protection (P0) must always be resolved before overkill reduction (P2).

If the script violates either rule, rewrite it before proceeding.

---
## CHECK 2B2 — Dynamic FP Penalty for High-Overkill Refinements

If the script is a refinement intended to reduce high overkill, or if it mentions
`best_overkill_rate`, `fp_penalty_loss`, `weighted_loss`, or `false_positive`,
call `static_fp_penalty_check_fn`.

For high-overkill refinements, the script must contain executable dynamic
false-positive penalty logic, not just comments:

  fp_weight = 1.0 + 5.0 * max(0, best_overkill_rate - 0.08)

The penalty must be applied to a false-positive / G-as-NG loss term. If the check
fails, rewrite the script before proceeding.

---
## CHECK 2C — Diagnostic Output Contract (Static Analysis, no execution)

First call `static_contract_check_fn` with the script.

Read the (possibly rewritten) script and verify it contains full-run diagnostic
output for all required markers:

  PROBE_METRICS: {{"ng_recall": ..., "overkill_rate": ..., "G_prob_mean": ..., "NG_prob_mean": ..., "should_continue": true/false}}
  CALIBRATION_STATS: {{"G_prob_mean": ..., "G_prob_std": ..., "NG_prob_mean": ..., "NG_prob_std": ...}}
  THRESHOLD_CURVE: [...]
  PREDICTIONS: [...]

If any marker is missing, rewrite the script before proceeding. `PROBE_METRICS`
is mandatory because the evaluator uses it to reject clearly bad candidates
before a full training run; the script should print it after a cheap probe and
exit early when `should_continue` is false. `CALIBRATION_STATS`
is mandatory because the deterministic diagnosis scorer needs G/NG probability
means to classify high-overkill failures reliably.

---
## CHECK 2C2 — Degenerate Threshold Guard (Static Analysis, no execution)

Call `static_degenerate_threshold_guard_check_fn` with the script.

Read the (possibly rewritten) script and verify it contains executable code after
threshold selection that detects and reports the degenerate low-threshold/high-overkill
failure case:

  if threshold <= 0.15 and overkill_rate > 0.50:
      print("DEGENERATE_THRESHOLD_WARNING: ...")

Equivalent variable names are acceptable, but the script must check both conditions
and print `DEGENERATE_THRESHOLD_WARNING`. If the check fails, rewrite the script to
add this guard before proceeding. This catches all-NG / near-all-NG operating points
that can show high recall while false-rejecting most G samples.

---
## CHECK 2F — Feature-Level Siamese Difference (Static Analysis, no execution)

This check is CONDITIONAL. Always call `static_difference_feature_check_fn` with the
script. It returns `applies` plus a structural AST report.

- If `applies` is False, the script did not opt into the feature-level Siamese
  difference family. Skip this check — do NOT add a difference feature; many valid
  candidates use other architectures.

- If `applies` is True, the script claims to use a feature-level shared-weight
  Siamese difference. This is a DIFFERENT thing from the existing pixel-level
  9-channel input: the 9-channel input concatenates `abs(img_l - img_r)` on the raw
  images BEFORE the encoder. The feature-level difference instead runs a single
  SHARED encoder on each image to get `f_L` and `f_R`, computes `abs(f_L - f_R)` on
  those FEATURES, and feeds the classifier head `concat([f_L, f_R, |f_L - f_R|])`
  (width == 3 x feature_dim). Both may coexist, but a feature-diff claim is only
  satisfied by feature-level code.

  Require ALL of the following from the report (these come from AST dataflow, not
  string matching):
    - `shared_encoder` is True (one `self.<attr>` applied to both images),
    - `independent_backbones` is False (NOT two separate backbones for L and R),
    - `abs_diff_of_features` is True (abs of the two encoder outputs),
    - `diff_and_both_features_in_concat` is True (concat contains f_L, f_R, and diff).

  If the report's `ok` is False, rewrite the model so it uses ONE shared encoder
  instance for both branches and feeds the head `concat([f_L, f_R, |f_L - f_R|])`
  with the head's first `nn.Linear` in_features set to `3 * feature_dim`. Then call
  `static_difference_feature_check_fn` again on the rewritten script. Do not accept a
  feature-diff candidate whose report `ok` is False.

---
## CHECK 2G — Small-Data-Safe Strategy Guard (Static Analysis, no execution)

Call `static_small_data_strategy_check_fn` with the script. This check is AST/static,
not prose-only.

The AOI grouped train split has only about 287 samples. The validator must flag and
rewrite candidates with any of these risks:
- large added parameter capacity with no regularization: no backbone freeze,
  no partial freeze, no dropout, and no nonzero weight_decay;
- legacy `checkpoints/data_split.json` usage instead of `checkpoints/data_split_grouped.json`;
- missing `roc_auc`, `prob_gap`, or `THRESHOLD_CURVE` reporting;
- missing degenerate/flat-prediction warning logic such as score-range,
  unique-score, or `DEGENERATE_PREDICTION_WARNING` / `DEGENERATE_THRESHOLD_WARNING`;
- repeat of known-failed fingerprints: mg7 threshold-only / threshold-acceptance
  tuning, mg8 global feature-difference-only, larger backbone, or
  two-independent-backbone stereo.
- full-freeze-only with no final-block/layer4 unfreeze is a warning. If the
  script or diagnosis evidence shows flat predictions or `prob_gap` collapse
  around 0.02, rewrite to freeze early backbone layers and unfreeze only the
  final ResNet block/layer4 or equivalent final layer group.

Augmentation safety rule: geometric augmentations must be applied IDENTICALLY to L
and R by sampling transform parameters once and applying them to both images. Reject
heavy color jitter, random erasing over defects, random perspective, or any
L/R-desynchronizing affine/rotation/crop. AOI-safe augmentation should be light,
paired, and local-contrast-preserving.

If the report's `ok` is False, rewrite the script before execution. Prefer, in this
order: partial-unfreeze last ResNet block/layer4 with a small regularized head when
full freeze underfits; weight decay + dropout; AOI-safe paired augmentation; smaller/
regularized head; localized patch/ROI or local L/R difference evidence. Calibration/
threshold curves are REPORTING, not the only fix.

---
## CHECK 2D — Per-Sample Prediction Output (Static Analysis, no execution)

Read the (possibly rewritten) script and verify that it prints at least one of the following
marker lines to stdout during a full (non-dry-run) training run:

  PREDICTIONS: [{{"sample_id": "img_001_L.bmp", "true_label": "NG", "predicted_label": "G", "ng_probability": 0.23}}, ...]
  ERROR_ANALYSIS: {{"fp_samples": [...], "fn_samples": [...], "available": true, ...}}

Field name requirements for PREDICTIONS entries:
  - sample identifier: "sample_id" (or "id" / "image_id" / "pair_id" / "img_l" / "path")
  - ground-truth label: "true_label" (or "label" / "y_true") — value must be "G" or "NG"
  - predicted label: "predicted_label" (or "prediction" / "y_pred") — value must be "G" or "NG"
  - NG probability: "ng_probability" (or "ng_prob" / "probability" / "score") — float 0–1

For ERROR_ANALYSIS the top-level keys must be "fp_samples" and "fn_samples" (NOT "fp"/"fn").

These markers allow the evaluator to collect per-sample FP/FN evidence for the
error-analysis workflow. If neither marker is present, add code to print one of them
after threshold selection.

IMPORTANT: This output MUST be guarded with `if not DRY_RUN:` so it is suppressed during
dry-run execution (CHECK 3 runs in DRY_RUN mode). CHECK 3 does NOT require these markers
in its stdout — their absence in dry-run output is expected and correct.

---
## CHECK 2E — Reproducibility Seeding (Static Analysis, no execution)

Read the script and verify that it reads the random seed from environment variables
and seeds ALL of the following before any model or data loader is constructed:
  - Python `random.seed(...)`
  - `numpy.random.seed(...)` or `np.random.seed(...)`
  - `torch.manual_seed(...)`

The seed value must come from `os.environ.get("AOI_RANDOM_SEED", ...)` or
`os.environ.get("SEED", ...)` — a hardcoded integer constant is NOT acceptable
because the evaluator injects different seeds for multiseed confirmation runs.

If any of these seeding calls are absent or use a hardcoded literal instead of the
environment variable, rewrite the script to add the correct seeding block:

  ```python
  import os, random
  _seed = int(os.environ.get("AOI_RANDOM_SEED", os.environ.get("SEED", 42)))
  random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)
  ```

This block must appear before any `DataLoader`, `Dataset`, model constructor, or
`train_test_split` call.

---
## CHECK 3 — Dry-Run Execution

The script will be executed in DRY_RUN mode with these environment variables already set:
  DRY_RUN=1, DRY_RUN_EPOCHS=1, DRY_RUN_SAMPLES=10

The script is expected to honour these by capping training to 1 epoch and using only
10 samples. This validates imports, shapes, and METRICS output — NOT training quality.

Call `run_script_fn` to execute the script.

If the script fails (returncode != 0, or "Traceback" appears in stderr, or timed_out is True):
  1. Diagnose the error from stderr / stdout.
  2. Rewrite the script to fix the root cause.
  3. Call `run_script_fn` again with the rewritten script.
  4. Repeat for up to {config.DEBUGGER_RETRY_CAP} total execution attempts.

If all {config.DEBUGGER_RETRY_CAP} attempts fail:
  - Call `append_failed_script_fn` with:
      script_name = the name you infer from the script (or "script" if unknown)
      error       = the last stderr / error message (truncated to 500 chars)
      attempts    = {config.DEBUGGER_RETRY_CAP}
  - End your response with exactly: VALIDATION_FAILED

If the script executes successfully (returncode == 0) AND the output contains "METRICS:":
  - End your response with the marker VALIDATED_SCRIPT: followed immediately (on the same line or the next line) by the complete, final validated script.
  - Do NOT add any prose after the script.

If the script exits 0 but does NOT print "METRICS:":
  - Treat this as a failure: the script did not produce required output.
  - Rewrite to ensure it prints METRICS: {{...}} as the last line and retry.

Example output on success:
VALIDATED_SCRIPT:
<full python script here>

Example output on failure after retry cap:
VALIDATION_FAILED
"""

code_validator_agent = LlmAgent(
    name="code_validator_agent",
    model=config.MODEL,
    description=(
        "Validates AOI training scripts for data leakage, stereo image usage, "
        "and runtime correctness (dry-run only). Retries with fixes up to the debug retry cap."
    ),
    instruction=_INSTRUCTION,
    tools=[
        static_contract_check_tool,
        static_fp_penalty_check_tool,
        static_degenerate_threshold_guard_check_tool,
        static_difference_feature_check_tool,
        static_small_data_strategy_check_tool,
        run_script_tool,
        _append_failed_script_tool,
    ],
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)

# Single guard interface consumed by all script-generating agents
code_validator_tool = agent_tool.AgentTool(agent=code_validator_agent)

# ---------------------------------------------------------------------------
# Validation cache tools
# Coder agents call store_validation_cache_fn after validating a script.
# Evaluator agents call check_validation_cache_fn before re-validating the
# same script, skipping the redundant validator call on a cache hit.
# ---------------------------------------------------------------------------


def _script_hash(script: str) -> str:
    return hashlib.sha256(script.encode()).hexdigest()


def check_validation_cache_fn(tool_context, script: str) -> str:
    """
    Check whether this script was already validated in the current session.
    Returns "CACHE_HIT: VALIDATED", "CACHE_HIT: VALIDATION_FAILED", or "CACHE_MISS".
    """
    h = _script_hash(script)
    cache = tool_context.state.get("_validation_cache") or {}
    if h in cache:
        return f"CACHE_HIT: {cache[h]}"
    return "CACHE_MISS"


def store_validation_cache_fn(tool_context, script: str, status: str) -> str:
    """
    Record the validation outcome for a script so evaluator agents can skip
    re-validation. Call with status="VALIDATED" or status="VALIDATION_FAILED".

    The cache is capped at config.VALIDATION_CACHE_MAX entries (FIFO by insertion
    order) so it cannot grow unboundedly across loop iterations (§6.7).
    """
    h = _script_hash(script)
    cache = dict(tool_context.state.get("_validation_cache") or {})
    # Refresh recency: re-insert at the end so the newest entries survive pruning.
    cache.pop(h, None)
    cache[h] = status
    while len(cache) > config.VALIDATION_CACHE_MAX:
        oldest = next(iter(cache))
        cache.pop(oldest, None)
    tool_context.state["_validation_cache"] = cache
    return f"Validation result '{status}' cached for script hash {h[:8]}."


check_validation_cache_tool = FunctionTool(func=check_validation_cache_fn)
store_validation_cache_tool = FunctionTool(func=store_validation_cache_fn)
