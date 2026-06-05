import json
import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.guards.code_validator_agent import code_validator_tool, store_validation_cache_tool
from mle_star_agent.shared.analytical_state import analytical_state_line
from mle_star_agent.shared.callbacks import (
    count_tokens_callback,
    rate_limit_retry_callback,
)
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint
from mle_star_agent.phases.phase2_refinement import fusion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load-on-demand context tool
# Pulls everything the coder needs into THIS turn instead of relying on the
# replayed session history. This is the load-on-demand pattern from
# ablation_agent.get_best_pipeline_script_fn, extended to the full coder input
# bundle so the agent can run with include_contents="none".
# ---------------------------------------------------------------------------


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


def _per_sample_evidence(state: dict) -> dict:
    """
    Compact per-sample FP/FN evidence from state["latest_error_analysis"] so the
    coder can see which probability range contains the false positives. Samples are
    capped; the full set stays on disk. Empty dict if no evidence is available.
    """
    evidence = state.get("latest_error_analysis")
    if not isinstance(evidence, dict) or not evidence.get("available"):
        return {}
    cap = config.ERROR_ANALYSIS_SAMPLE_CAP
    fp = list(evidence.get("fp_samples", []) or [])
    fn = list(evidence.get("fn_samples", []) or [])
    return {
        "fp_count": evidence.get("fp_count"),
        "fn_count": evidence.get("fn_count"),
        "probability_summary": evidence.get("probability_summary", {}),
        "fp_samples": fp[:cap],
        "fn_samples": fn[:cap],
    }


def load_refinement_context_fn(tool_context) -> str:
    """
    Pull everything the refinement coder needs into THIS turn (not history).

    Call this FIRST, before planning or drafting the script. Returns the current
    best pipeline script plus the diagnosis, error-analysis, selected strategy,
    and the current best metrics. Replaces reading these values from conversation
    context so the agent works with include_contents="none".
    """
    s = tool_context.state
    script = s.get("best_pipeline_script", "")
    # Disk fallback if the state key is absent (crash recovery, retry start).
    if not script and checkpoint_exists(config.CKPT_BEST_PIPELINE):
        try:
            script = load_checkpoint(config.CKPT_BEST_PIPELINE).get("best_pipeline_script", "")
        except RuntimeError:
            logger.warning("best_pipeline.json unreadable during context load; coder will build from scratch.")

    diag = s.get("diagnosis_report", {}) or {}
    if not isinstance(diag, dict):
        diag = {}
    err = s.get("error_analysis_report", {}) or {}
    if not isinstance(err, dict):
        err = {}

    # Modality-conditional guidance. Default "stereo" preserves all existing behaviour
    # (the static _INSTRUCTION already carries the stereo strategy list). For mono we
    # inject an authoritative override that omits stereo-only strategies and the
    # paired-augmentation rule, and substitutes single-image strategies.
    input_modality = s.get("input_modality", "stereo")
    if input_modality == "mono":
        modality_block = (
            "INPUT_MODALITY: mono\n"
            "MODALITY OVERRIDE (authoritative — overrides any stereo wording in your instructions):\n"
            "- This dataset has a SINGLE image per sample (key `img`, absolute path); there is no "
            "`img_l`/`img_r` pair. Load the single `img` as a standard 3-channel input.\n"
            "- DO NOT use stereo strategies: cross_attention_stereo, global feature difference, "
            "shared-weight Siamese difference, 9-channel L/R/diff input, or the FEATURE_DIFF_CANDIDATE "
            "marker. They are not applicable to mono input.\n"
            "- IGNORE the 'apply geometric transforms identically to L and R' augmentation rule — "
            "there is no pair to keep in sync.\n"
            "- Prefer single-image separability strategies instead: spatial attention on backbone "
            "features, mixup/CutMix, test-time augmentation (TTA), and patch dropout."
        )
    else:
        modality_block = "INPUT_MODALITY: stereo"

    parts = [
        analytical_state_line(s),
        modality_block,
        f"OUTER_ITERATION: {s.get('outer_iteration', 0)}  INNER_ITERATION: {s.get('inner_iteration', 0)}",
        f"SELECTED_STRATEGY:\n{s.get('selected_refinement_strategy', '')}",
        f"SELECTED_STRATEGY_FINGERPRINT:\n{json.dumps(s.get('selected_strategy_fingerprint', {}), default=str)}",
        f"STRATEGY_SELECTION_REASON:\n{s.get('strategy_selection_reason', '')}",
        f"STRATEGY_CANDIDATES:\n{json.dumps(s.get('refinement_strategy_candidates', {}), default=str)}",
        f"TARGET_COMPONENT: {diag.get('target_component', '')}",
        f"RECOMMENDED_CHANGES:\n{diag.get('recommended_changes', '')}",
        f"IMPACT_SUMMARY:\n{diag.get('impact_summary', '')}",
        f"FAILURE_CLASSIFICATION: {json.dumps(diag.get('failure_classification'), default=str)}",
        f"PREDICTION: {json.dumps(diag.get('prediction'), default=str)}",
        "ERROR_ANALYSIS_REPORT: "
        f"dominant_failure={err.get('dominant_failure')} | "
        f"threshold_fix_possible={err.get('threshold_fix_possible')} | "
        f"recommended_target_component={err.get('recommended_target_component')} | "
        f"recommended_changes={err.get('recommended_changes')} | "
        f"evidence_summary={err.get('evidence_summary')}",
        f"PER_SAMPLE_ERROR_EVIDENCE (capped; full set on disk): "
        f"{json.dumps(_per_sample_evidence(s), default=str)}",
        f"BEST_OVERKILL_RATE: {s.get('best_overkill_rate')}",
        f"BEST_MISS_RATE: {s.get('best_miss_rate')}",
        f"LATEST_CALIBRATION_STATS: {json.dumps(s.get('latest_calibration_stats', {}), default=str)}",
        f"LATEST_PROBE_METRICS: {json.dumps(s.get('latest_probe_metrics', {}), default=str)}",
        f"LATEST_EPOCH_LOGS: {json.dumps(s.get('latest_epoch_logs', []), default=str)}",
        f"REFINEMENT_POPULATION (summary, no scripts): "
        f"{json.dumps(_population_summary(s.get('refinement_population', [])), default=str)}",
        f"INSTRUMENTATION_REQUIRED: {bool(s.get('error_analysis_instrumentation_required'))}",
        f"BEST_PIPELINE_SCRIPT:\n{script}" if script
        else "BEST_PIPELINE_SCRIPT: (none — this is the first refinement; build from scratch per the strategy)",
    ]
    return "\n\n".join(parts)


_load_context_tool = FunctionTool(func=load_refinement_context_fn)


def load_fusion_scripts_fn(tool_context) -> str:
    """
    Fetch the FULL text + metrics of the top-2 refinement_population members for a
    cross-branch FUSION attempt (load-on-demand, like get_best_pipeline_script).

    Call this ONLY when SELECTED_STRATEGY starts with the fusion marker
    (`cross_branch_fusion`). Returns FUSION_SCRIPT_0 (BASE) and FUSION_SCRIPT_1
    (DONOR) with their metrics, prefixed by the mono/stereo modality contract.
    Returns an error string (never raises) if fewer than 2 members are archived.
    """
    members = fusion.top_fusion_members(tool_context.state, k=2)
    if len(members) < 2:
        return (
            "FUSION_SCRIPTS_UNAVAILABLE: refinement_population has fewer than 2 "
            "members — implement the selected strategy as a normal single-script change."
        )
    return fusion.render_fusion_scripts(tool_context.state, members)


_load_fusion_scripts_tool = FunctionTool(func=load_fusion_scripts_fn)

# ---------------------------------------------------------------------------
# Refinement plan FunctionTool
# Called BEFORE the agent writes its final script output, so the plan is
# committed to state while the LLM still has context to produce the script.
# ---------------------------------------------------------------------------


def save_refinement_plan_fn(
    tool_context,
    target_component: str,
    changes_summary: str,
    implementation_steps: str,
) -> str:
    """
    Write the refinement plan to state["refinement_plan"] before the agent
    generates the final script.  Called once per inner iteration.

    Args:
        target_component: The pipeline component being improved (from diagnosis_report).
        changes_summary: One-sentence description of the change being made.
        implementation_steps: Numbered list of concrete code-level changes as a string.
    """
    n = int(tool_context.state.get("outer_iteration", 0))
    m = int(tool_context.state.get("inner_iteration", 0))

    plan = {
        "outer_iteration": n,
        "inner_iteration": m,
        "target_component": target_component,
        "changes_summary": changes_summary,
        "implementation_steps": implementation_steps,
    }
    tool_context.state["refinement_plan"] = plan

    logger.info(
        "Refinement plan saved (outer=%d, inner=%d, target='%s')",
        n, m, target_component,
    )
    return (
        f"Refinement plan saved for outer={n}, inner={m}. "
        f"Target: '{target_component}'. "
        "Now output the complete refined Python script as your FINAL response."
    )


_save_plan_tool = FunctionTool(func=save_refinement_plan_fn)

# ---------------------------------------------------------------------------
# Agent instruction
# IMPORTANT: output_key="current_script" captures the agent's last text
# response verbatim.  The final response MUST be the complete Python script
# and nothing else — no commentary, no markdown fences, no trailing text.
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are the Refinement Coder Agent.

Your role is to improve the current best pipeline script based on the diagnosis
report from the ablation phase, then produce the updated script for evaluation.

---
## STEP 1 — Load inputs

Call `load_refinement_context_fn` FIRST. It returns everything you need for this
iteration in a single block — do NOT rely on conversation history for these values.
The returned block contains:
- BEST_PIPELINE_SCRIPT — the current best training script (full, verbatim). If it
    says "(none ...)", this is the first refinement; build from scratch per the strategy.
- TARGET_COMPONENT / RECOMMENDED_CHANGES / IMPACT_SUMMARY / FAILURE_CLASSIFICATION /
    PREDICTION — from the diagnosis report. PREDICTION is a falsifiable set of metric
    constraints your implementation should be able to satisfy.
- ERROR_ANALYSIS_REPORT — if present from the previous evaluation, treat it as concrete
    FP/FN evidence: dominant_failure, threshold_fix_possible, evidence_summary,
    recommended_target_component, recommended_changes.
- SELECTED_STRATEGY — the specific strategy selected by the Refinement Planner Agent
    for this inner iteration. This is the primary instruction for what to implement.
    STRATEGY_SELECTION_REASON explains why; STRATEGY_CANDIDATES lists all candidates.
    **Implement exactly the selected strategy — do not substitute a different one.**
- SELECTED_STRATEGY_FINGERPRINT — stable dedupe labels with target_component and
    mechanism_class. When target_component is `optimizer/lr-schedule`, the
    mechanism_class names the exact optimizer/scheduler combo to implement.
- REFINEMENT_POPULATION (summary) — metrics of accepted improvements and lower-overkill
    non-best candidates. Preserve concrete FP-control mechanisms from lower-overkill
    candidates while keeping the current best pipeline as the primary starting point.
- LATEST_CALIBRATION_STATS — G_prob_mean, NG_prob_mean etc. from last training run.
- LATEST_PROBE_METRICS — if present, cheap probe evidence that may identify the next
    target component.
- BEST_OVERKILL_RATE — current best overkill rate. If > 0.08, high-overkill control
    is mandatory.
- BEST_MISS_RATE — current best miss rate (P0 gate; see FP-penalty rules below).
- LATEST_EPOCH_LOGS — per-epoch train/val metrics from last training run.
- INSTRUMENTATION_REQUIRED — if True, the previous script failed to emit PREDICTIONS or
    ERROR_ANALYSIS output. **This overrides the selected strategy: your top priority this
    iteration is to ensure the script emits all required output lines.** Apply the
    selected strategy AND fix instrumentation.
- OUTER_ITERATION / INNER_ITERATION — loop counters.

---
## STEP 2 — Plan the changes

**If `state["error_analysis_instrumentation_required"]` is True:**
The previous script did not emit PREDICTIONS or ERROR_ANALYSIS. You MUST add these
output lines to the script this iteration in addition to the selected strategy change.
Without this output the gate will continue to flag missing evidence.

If `state["selected_refinement_strategy"]` starts with `preflight_probe` or the
selected strategy fingerprint target/mechanism is `preflight_probe`, write a
probe-only script. It should run cheap diagnostics, print exactly one
`PROBE_METRICS: {...}` line with `probe_only: true`, `should_continue: false`,
`ng_recall`, `overkill_rate`, `G_prob_mean`, `NG_prob_mean`, `probability_gap`,
and `recommended_target_component`, then exit before full training. This is not
a failed model; it is evidence collection for the next planner pass.

If `state["selected_refinement_strategy"]` starts with `cross_branch_fusion`, this
is a FUSION attempt, not a normal single-script change. Call `load_fusion_scripts_fn`
to fetch the two full population scripts (FUSION_SCRIPT_0 = BASE, FUSION_SCRIPT_1 =
DONOR) with their metrics. Keep the BASE as the skeleton (one model, one training
loop) and transplant ONLY the single block where the DONOR is measurably better
(its loss / augmentation / calibration / threshold logic) — do NOT stitch two
incompatible backbones. If the two are architecturally incompatible to fuse cleanly,
transplant only the donor's architecture-agnostic data/loss/calibration/threshold
block onto the base. Honor the INPUT_MODALITY line in both the directive and the
fetched scripts EXACTLY — never introduce stereo code for mono input. The fused
script must print the standard METRICS line and all the usual diagnostic lines.

Otherwise, implement `state["selected_refinement_strategy"]` exactly as described.
Translate the strategy name and description into specific code modifications. Consider:
- What existing code blocks implement the target component?
- What is the minimal, focused change that applies this strategy?
- Does the change interact with any other component (data loading, loss, metrics)?
- Does the change reduce false positives / overkill while preserving or moving toward
  FN=0? A higher-recall model with high overkill is not progress.
- Use PER_SAMPLE_ERROR_EVIDENCE (from `load_refinement_context_fn`) to guide the
  exact implementation details — its `probability_summary` and `fp_samples` show
  which probability range contains the FPs. Combine with ERROR_ANALYSIS_REPORT's
  interpreted recommendation.
- If current metrics show high overkill, include explicit error analysis in the script:
  collect per-sample NG probabilities, threshold, predicted label, true label, and
  sample identifiers for every FP and FN. Print a compact ERROR_ANALYSIS JSON line after
  METRICS so the next diagnosis can inspect the actual mistakes.
- Before full training, run a cheap probe on the capped/early training signal and print
  `PROBE_METRICS: {...}`. Include at least `ng_recall`, `overkill_rate`, `G_prob_mean`,
  `NG_prob_mean`, and `should_continue`. Only set `should_continue: false` for
  **catastrophic** failures sustained across ALL probe epochs: overkill > 0.90 at
  threshold 0.5 (model classifies nearly all G as NG), recall collapse ng_recall < 0.05
  at threshold 0.5 (model misses nearly all NG), or truly flat predictions
  `abs(NG_prob_mean - G_prob_mean) < 0.01` (gap so small no threshold can separate).
  Do NOT abort on borderline overkill (0.50–0.90) or borderline prob_gap (0.01–0.05)
  — those cases often recover during full training with the weighted loss and threshold sweep.
- If `state["best_overkill_rate"] > 0.08` **AND** `state["best_miss_rate"] <= 0.03`,
  the refined script MUST implement dynamic false-positive loss control as executable
  code: `fp_weight = 1.0 + 5.0 * max(0, best_overkill_rate - 0.08)`. Apply this FP
  weight to a G-as-NG / false-positive penalty term while preserving the existing
  weighted loss.
  **CRITICAL GATE — do NOT apply FP penalty when `best_miss_rate > 0.03`.** FP penalty
  makes the model more conservative → it predicts fewer NGs → miss_rate worsens. When
  both P0 (miss) and P2 (overkill) are failing simultaneously, P0 takes strict priority:
  target recall first, address overkill only after miss_rate ≤ 0.03.
  **NULL SAFETY — if `state["best_miss_rate"]` is absent or None, treat it as 1.0
  (worst case). Do NOT apply FP penalty when the key is missing.**

If SELECTED_STRATEGY_FINGERPRINT.target_component is `optimizer/lr-schedule`,
make optimizer and learning-rate schedule a first-class, isolated refinement:
- Change only the optimizer/scheduler hyperparameters and their scheduler.step()
  placement. Preserve the model architecture, data split, threshold policy, loss
  family, DRY_RUN handling, `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20`, early
  stopping, PROBE_METRICS, EPOCH_LOG, CALIBRATION_STATS, THRESHOLD_CURVE,
  PREDICTIONS, and METRICS output.
- Implement the selected mechanism_class concretely:
  `adamw_cosine_restart_tune` -> AdamW with tuned lr/weight_decay and
  `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=..., T_mult=..., eta_min=...)`.
  `sgd_momentum_plateau` -> `torch.optim.SGD(..., lr=..., momentum=0.9,
  nesterov=True, weight_decay=...)` with `ReduceLROnPlateau(optimizer, mode="min",
  factor=..., patience=..., min_lr=...)` stepped as `scheduler.step(val_loss)`.
  `adamw_plateau_decay` -> AdamW with tuned lr/weight_decay and
  `ReduceLROnPlateau` stepped on validation loss.
  `warm_restart_*` -> keep the existing optimizer family unless the selected
  strategy explicitly says otherwise, raise the base LR back up for one SGDR-style
  cycle with `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`, and optionally
  re-initialize only the final classification head if it is clearly isolated.
- Add a simple script constant near the optimizer block, for example
  `OPTIMIZER_LR_SCHEDULE_VARIANT = "sgd_momentum_plateau"`, so the attempted
  combo is visible in logs and diffs. Do not emit this instead of METRICS; it is
  just an in-script label.
- A fixed-LR script is invalid for this target. The Fix #1 scheduler validator
  will hard-reject variants without a real PyTorch scheduler and correct
  scheduler.step() call.

small-data-safe strategy policy for the current grouped train split (~287 samples):
prefer, in priority order: (1) freeze or partially-freeze the pretrained backbone
+ small head, with partial-unfreeze adaptation preferred when full freeze underfits:
freeze early backbone layers, unfreeze only the final ResNet block/layer4 or equivalent,
and use a small regularized head; (2) weight decay + dropout; (3) AOI-safe augmentation;
(4) smaller/regularized head; (5) local/patch or localized L/R difference evidence.
Calibration/threshold curves are REPORTING, not the only fix. They must be emitted
for diagnosis, but a threshold-only change is not an adequate modeling fix after
mini-goal 7. De-prioritize and avoid: larger backbone; two-independent-backbone stereo;
global feature-difference-only; known failed mg7 threshold-only / threshold-acceptance
fingerprints; known failed mg8 global `|f_L-f_R|` fingerprint; full-freeze-only
strategy if predictions are flat or `prob_gap` collapses near zero.
For augmentation, sample geometric parameters once and apply them IDENTICALLY to L
and R. Do not use heavy color jitter, random erasing over defects, random perspective,
or L/R-desynchronizing affine/rotation/crop.

---
## STEP 3 — Save the refinement plan

Call `save_refinement_plan_fn` with:
- `target_component`    : the active component for this iteration. Prefer
                          SELECTED_STRATEGY_FINGERPRINT.target_component when
                          present. If absent, use error_analysis_report.
                          recommended_target_component, otherwise diagnosis_report
                          target_component. For optimizer and scheduler tuning,
                          this must be exactly `optimizer/lr-schedule`.
- `changes_summary`     : begin with the strategy name from state["selected_refinement_strategy"],
                          then one sentence describing the implementation (e.g.
                          "cross_attention_stereo: Replace channel-concat stereo fusion with cross-attention block")
- `implementation_steps`: numbered list of specific code changes as a single string

---
## STEP 4 — Draft the refined script

Starting from the BEST_PIPELINE_SCRIPT returned by `load_refinement_context_fn`, apply
ONLY the planned changes to the target_component block.  Do NOT change unrelated components.

The script must:
- Load both _L and _R stereo images (unless a prior ablation proved stereo is not useful)
- Load Excel labels
- Train with weighted loss (unless explicitly changing this)
- When `state["best_overkill_rate"] > 0.08` **AND** `state["best_miss_rate"] <= 0.03`,
  include executable dynamic FP penalty code:
  `fp_weight = 1.0 + 5.0 * max(0, best_overkill_rate - 0.08)` applied to a
  false-positive loss term. It must be executable code, not only a comment.
  Skip this entirely when `best_miss_rate > 0.03` — FP penalty conflicts with P0.
  Skip this entirely when `state["best_miss_rate"]` is absent or None (treat as 1.0).
- Sweep threshold on the validation set (unless explicitly changing this)
- Threshold selection must use strict two-stage priority:
  Stage 0 — filter out thresholds with `FP > 2` (`FP <= 2` is mandatory for the 30-G
  validation/test scale). If no threshold survives this filter, choose the threshold
  with the minimum FP count, then lowest miss_rate, and report it as below-target.
  Stage 1 — find the threshold that minimises miss_rate (P0 target: <= 0.03).
  Stage 2 — among ALL thresholds tied for that minimum miss_rate, pick the one
  with the lowest overkill_rate (P2 target: <= 0.08).
  Do NOT use acceptance-distance averaging or a blended score — this blurs the
  P0/P2 priority order. Miss_rate must be resolved first, then overkill minimised.
- **Probability calibration before the sweep** (MANDATORY): after training, fit
  `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on the VALIDATION
  scores (`X = raw_val_ng_scores`, `y = val_true_binary` with 1=NG, 0=G), then map
  all val/test scores through the fitted calibrator
  (`cal_probs = iso.transform(raw_scores)`) so every probability is on the
  calibrated [0,1] scale. The two-stage threshold selection above, the
  THRESHOLD_CURVE, the final test metrics, and the `threshold` reported in METRICS
  MUST ALL be computed on these CALIBRATED probabilities — never the raw scores.
  Labels are binary G/NG only; there are no defect sub-classes, so this is a single
  global calibrator and a single global threshold.
- **Fine-grained sweep step** (MANDATORY): sweep the threshold across 0.10–0.90
  inclusive at step **0.01** (not 0.05) — `[round(0.10 + 0.01 * i, 2) for i in range(81)]`.
  The finer grid lets selection land on operating points (e.g. 0.37) the old 0.05
  grid skipped, which is how miss_rate is pushed toward the <= 0.03 target.
- When overkill remains high, add AOI-specific separability improvements before another
  generic backbone swap: ROI/contrast normalization, L/R alignment checks, absolute
  difference maps, SSIM-like difference features, local defect patches, probability
  calibration, or a constrained loss/threshold objective.
- If the previous evidence shows flat predictions, collapsed `prob_gap` around 0.02,
  or underfit from a fully frozen backbone, do NOT retry full-freeze-only. Freeze
  early layers and unfreeze only the final ResNet block/layer4 (or equivalent final
  layer group) with a small dropout head, AdamW, and nonzero weight_decay.
- **Feature-level shared-weight Siamese difference** is an available separability
  strategy and is DIFFERENT from the pixel-level 9-channel `abs(img_l - img_r)` input.
  Pixel-level diff concatenates the raw-image difference BEFORE the encoder. The
  feature-level version runs ONE shared encoder on each image to get `f_L` and `f_R`,
  computes `abs(f_L - f_R)` on the FEATURES, and feeds the head
  `concat([f_L, f_R, |f_L - f_R|])` (head first `nn.Linear` in_features == 3 × feat_dim).
  If the SELECTED_STRATEGY targets this, add the marker `FEATURE_DIFF_CANDIDATE = True`
  near the top, use a SINGLE encoder instance for both branches (shared weights, never
  two independent backbones), and keep all else fixed so the change is isolated. The
  validator's feature-difference check (AST + structural) will reject a comment-only or
  two-backbone implementation.
- Apply data augmentation (unless explicitly changing this)
- Print exactly: METRICS: {"accuracy": ..., "ng_recall": ..., "miss_rate": ...,
  "overkill_rate": ..., "f1": ..., "avg_latency_ms": ..., "threshold": ...,
  "ng_count": ..., "g_count": ..., "tp": ..., "tn": ..., "fp": ..., "fn": ...,
  "roc_auc": ..., "prob_gap": ...}
  `threshold` is a single float and MUST be the selected operating point on the
  CALIBRATED-probability scale (the output of the isotonic calibrator), not the raw
  score scale.
  where `roc_auc = sklearn.metrics.roc_auc_score(y_true_binary, ng_probs)` on the test set
  (y_true_binary: 1=NG, 0=G; emit 0.0 if only one class present), and
  `prob_gap = mean(ng_probs[true==NG]) - mean(ng_probs[true==G])` (positive = good separability)
- Load train/val/test paths and sample IDs from `checkpoints/data_split_grouped.json` (the grouped split is the default since mini-goal 7; do NOT use the legacy `checkpoints/data_split.json`) — the script runs as a standalone process with no ADK state access. Use: `import json; data_split = json.load(open("checkpoints/data_split_grouped.json"))`
- Seed ALL random number generators from the environment before any model or data loader
  initialization (this is mandatory — the evaluator injects AOI_RANDOM_SEED for reproducibility):
  ```python
  import os, random
  _seed = int(os.environ.get("AOI_RANDOM_SEED", os.environ.get("SEED", 42)))
  random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)
  ```
- You MUST set `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20` with early stopping patience 3 epochs based on validation loss. Do NOT hardcode 5 — the line must read exactly `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20`.
- Use a real PyTorch learning-rate schedule instead of a fixed LR. Prefer either `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)` (SGDR) or `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)`. Instantiate the scheduler after the optimizer and call `scheduler.step()` correctly in the training loop: for `CosineAnnealingWarmRestarts`, step once per epoch (or batch with fractional epoch); for `ReduceLROnPlateau`, call `scheduler.step(val_loss)` after validation loss is computed. The validator will hard-reject schedule-less scripts.
- Respect `config.TIMEOUT_SECONDS` (7200 s / 2 hours) — keep the script fast enough to finish
- Must print `EPOCH_LOG: {{...}}` after each epoch (epoch, train_loss, val_loss, val_ng_recall, val_overkill)
- Must print `PROBE_METRICS: {{...}}` before full training (ng_recall, overkill_rate,
  G_prob_mean, NG_prob_mean, should_continue, reason)
- Must print `CALIBRATION_STATS: {{...}}` after training (G_prob_mean, G_prob_std, NG_prob_mean, NG_prob_std)
- Must print `THRESHOLD_CURVE: [...]` during val sweep (t, recall, overkill, miss_rate, accuracy per threshold)
- Must print `PREDICTIONS: [...]` after METRICS (sample_id, true_label, predicted_label, ng_probability, threshold for ALL test samples)

---
## STEP 5 — Validate the script

Call `code_validator_agent` with the complete drafted script text.
- If it returns "VALIDATED_SCRIPT:": extract the script that follows.
  Call `store_validation_cache_fn` with that extracted script and status "VALIDATED".
  The extracted script MUST be byte-for-byte identical to what you will output as
  your final response in STEP 6 — copy it exactly, do not paraphrase or reformat.
- If it returns "VALIDATION_FAILED": call `store_validation_cache_fn` with the script
  you passed to `code_validator_agent` and status "VALIDATION_FAILED".
  Use that same original script as your final output with a comment at the top noting the issue.

---
## STEP 6 — Output the script (CRITICAL)

Your FINAL response must be the complete Python script and NOTHING else.

Rules:
1. Start immediately with the first line of Python code (e.g. `import ...` or `# ...`).
2. Do NOT wrap the script in markdown code fences (no ```python ... ```).
3. Do NOT add any explanation, summary, or trailing text after the script.
4. The script must be complete and self-contained — no placeholders, no ellipsis.

This final response is captured verbatim as `state["current_script"]` and passed
directly to the evaluator.  Any non-Python text will cause execution to fail.
"""

# ---------------------------------------------------------------------------
# Refinement coder agent
# output_key captures the agent's last text response as state["current_script"]
# ---------------------------------------------------------------------------

refinement_coder_agent = LlmAgent(
    name="refinement_coder_agent",
    model=config.MODEL_PRO,
    description=(
        "Reads the diagnosis report and best pipeline script, plans and applies a "
        "targeted improvement to the identified component, validates the result, "
        "and outputs the complete refined script as state['current_script']."
    ),
    instruction=_INSTRUCTION,
    tools=[_load_context_tool, _load_fusion_scripts_tool, _save_plan_tool, code_validator_tool, store_validation_cache_tool],
    output_key="current_script",
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
