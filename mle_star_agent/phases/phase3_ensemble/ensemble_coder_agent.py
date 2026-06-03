import json
import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.guards.code_validator_agent import code_validator_tool, store_validation_cache_tool
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint
from mle_star_agent.shared.callbacks import (
    count_tokens_callback,
    log_context_size_callback,
    rate_limit_retry_callback,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load-on-demand context tool
# Pulls the full best pipeline script plus the compact ensemble signals into
# THIS turn instead of relying on replayed history, so the agent can run with
# include_contents="none".
# ---------------------------------------------------------------------------


def _candidate_scores_summary(state: dict) -> list[dict]:
    """Phase 1 baseline metrics only — never the candidate scripts themselves."""
    scores = state.get("candidate_scores", []) or []
    summary = []
    for entry in scores if isinstance(scores, list) else []:
        if not isinstance(entry, dict):
            continue
        summary.append({
            "name": entry.get("name"),
            "metrics": entry.get("metrics", {}),
        })
    return summary


def load_ensemble_context_fn(tool_context) -> str:
    """
    Pull everything the ensemble coder needs into THIS turn (not history).

    Call this FIRST, before choosing a strategy or drafting the script. Returns
    the best pipeline script (full, verbatim) plus the scalar ensemble signals,
    calibration stats, Phase 1 candidate metrics, ablation results, and diagnosis
    so the agent can run with include_contents="none".
    """
    s = tool_context.state
    script = s.get("best_pipeline_script", "")
    # Disk fallback if the state key is absent (crash recovery, retry start).
    if not script and checkpoint_exists(config.CKPT_BEST_PIPELINE):
        try:
            script = load_checkpoint(config.CKPT_BEST_PIPELINE).get("best_pipeline_script", "")
        except RuntimeError:
            logger.warning("best_pipeline.json unreadable during ensemble context load; using empty base.")

    diag = s.get("diagnosis_report", {}) or {}
    if not isinstance(diag, dict):
        diag = {}

    # Modality-conditional guidance. Default "stereo" preserves all existing behaviour.
    # For mono we inject an authoritative override omitting stereo-specific ensemble
    # instructions (e.g. "load both _L and _R images").
    input_modality = s.get("input_modality", "stereo")
    if input_modality == "mono":
        modality_block = (
            "INPUT_MODALITY: mono\n"
            "MODALITY OVERRIDE (authoritative — overrides any stereo wording in your instructions):\n"
            "- This dataset has a SINGLE image per sample (`img`); there is no `img_l`/`img_r` "
            "pair. Load the single `img` as a standard 3-channel input — ignore any instruction "
            "to load both _L and _R images or to use stereo fusion.\n"
            "- Build ensemble components from single-image backbones (different architectures, "
            "seeds, or augmentation regimes); do NOT use stereo-fusion variants as components."
        )
    else:
        modality_block = "INPUT_MODALITY: stereo"

    parts = [
        modality_block,
        f"ENSEMBLE_ITERATION: {s.get('ensemble_iteration', 0)}",
        f"CURRENT_BEST_SCORE (Phase 2 ng_recall): {s.get('current_best_score')}",
        f"ENSEMBLE_BEST_SCORE (so far in Phase 3): {s.get('ensemble_best_score')}",
        f"ENSEMBLE_BEST_OVERKILL: {s.get('ensemble_best_overkill')}",
        f"LATEST_CALIBRATION_STATS: {json.dumps(s.get('latest_calibration_stats', {}), default=str)}",
        f"CANDIDATE_SCORES (Phase 1 baselines, metrics only): "
        f"{json.dumps(_candidate_scores_summary(s), default=str)}",
        f"ABLATION_RESULTS: {json.dumps(s.get('ablation_results', []), default=str)}",
        "DIAGNOSIS_REPORT: "
        f"target_component={diag.get('target_component')} | "
        f"recommended_changes={diag.get('recommended_changes')}",
        f"BEST_PIPELINE_SCRIPT:\n{script}" if script
        else "BEST_PIPELINE_SCRIPT: (none — best_pipeline_script missing from state and disk)",
    ]
    return "\n\n".join(parts)


_load_context_tool = FunctionTool(func=load_ensemble_context_fn)


def _strategy_fingerprint(strategy_name: str, combination_method: str) -> str:
    return f"{strategy_name.strip().lower()}::{combination_method.strip().lower()}"


def _failed_ensemble_fingerprints(state: dict) -> set[str]:
    history = list(state.get("tried_ensemble_approaches", []) or [])
    if checkpoint_exists(config.CKPT_TRIED_ENSEMBLE_APPROACHES):
        checkpoint = load_checkpoint(config.CKPT_TRIED_ENSEMBLE_APPROACHES)
        history.extend(checkpoint.get("tried_ensemble_approaches", []) or [])

    failed = set()
    for entry in history:
        result = entry.get("result") or {}
        if result.get("improved") is False:
            failed.add(_strategy_fingerprint(
                entry.get("strategy_name", ""),
                entry.get("combination_method", ""),
            ))
    return failed

# ---------------------------------------------------------------------------
# Ensemble strategy FunctionTool
# Called BEFORE the agent writes its final script output so the strategy is
# committed to state while the LLM still has context to produce the script.
# ---------------------------------------------------------------------------


def save_ensemble_strategy_fn(
    tool_context,
    strategy_name: str,
    combination_method: str,
    component_descriptions: str,
) -> str:
    """
    Write the ensemble strategy to state["ensemble_strategy"] before the agent
    generates the final script.  Called once per ensemble iteration.

    Args:
        strategy_name: Short name for this ensemble approach (e.g. "weighted_avg_stereo_fusion").
        combination_method: How predictions are combined (e.g. "weighted average", "max voting",
                            "stacking", "threshold_ensemble").
        component_descriptions: Numbered list describing each model/component in the ensemble
                                and its role, as a single string.
    """
    n = int(tool_context.state.get("ensemble_iteration", 0))
    fingerprint = _strategy_fingerprint(strategy_name, combination_method)
    if fingerprint in _failed_ensemble_fingerprints(tool_context.state):
        return (
            "ENSEMBLE_STRATEGY_REJECTED: this strategy/method already failed. "
            "Choose a materially different ensemble strategy before writing code."
        )

    strategy = {
        "ensemble_iteration": n,
        "strategy_name": strategy_name,
        "combination_method": combination_method,
        "component_descriptions": component_descriptions,
        "strategy_fingerprint": fingerprint,
    }
    tool_context.state["ensemble_strategy"] = strategy

    logger.info(
        "Ensemble strategy saved (iteration=%d, strategy='%s', method='%s')",
        n, strategy_name, combination_method,
    )
    return (
        f"Ensemble strategy saved for iteration={n}. "
        f"Strategy: '{strategy_name}', method: '{combination_method}'. "
        "Now output the complete ensemble Python script as your FINAL response."
    )


_save_strategy_tool = FunctionTool(func=save_ensemble_strategy_fn)

# ---------------------------------------------------------------------------
# Agent instruction
# IMPORTANT: output_key="ensemble_script" captures the agent's last text
# response verbatim.  The final response MUST be the complete Python script
# and nothing else — no commentary, no markdown fences, no trailing text.
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are the Ensemble Coder Agent.

Your role is to design and implement an ensemble strategy that combines the best
pipeline script with complementary approaches to push performance beyond what
any single model can achieve.

---
## STEP 1 — Load inputs

Call `load_ensemble_context_fn` FIRST. It returns everything you need in a single
block — do NOT rely on conversation history for these values. The returned block
contains:
- BEST_PIPELINE_SCRIPT — the best single-model training script from Phase 2 (full,
  verbatim). Base each component model on it; modify only what the chosen strategy needs.
- CURRENT_BEST_SCORE — the ng_recall achieved by the best pipeline.
- ENSEMBLE_ITERATION — current ensemble attempt index (0-based).
- ENSEMBLE_BEST_SCORE — best ng_recall achieved so far in the ensemble phase
  (may be 0.0 on first iteration); ENSEMBLE_BEST_OVERKILL is its overkill.
- LATEST_CALIBRATION_STATS — single-model calibration distributions. Use this to
  intelligently average probabilities (e.g. temperature scaling or variance weighting)
  rather than blindly merging predictions, avoiding overkill cascade.
- CANDIDATE_SCORES — Phase 1 baseline metrics.
- ABLATION_RESULTS — component importance from Phase 2 ablation.
- DIAGNOSIS_REPORT — identified high-impact components.

---
## STEP 2 — Choose an ensemble strategy

Select a strategy based on the ensemble_iteration index to explore diverse approaches:

**Iteration 0** — Weighted-average ensemble:
  Train two complementary models (e.g. different stereo fusion methods or backbones),
  compute weighted average of their NG probability outputs, then sweep the combined
  threshold on the validation set.

**Iteration 1 — Second-stage verifier cascade**:
  Train or derive a high-recall first-stage detector from the best pipeline, then
  train a second-stage verifier that only reviews samples predicted NG by stage 1.
  The verifier's job is to rescue obvious G false positives while preserving NG
  recall. The final prediction should keep NG when either the sample is confidently
  NG or the verifier cannot confidently prove it is G; otherwise downgrade to G.

**Iteration 2** — Augmentation-diversity ensemble:
  Train the best pipeline three times with different random augmentation seeds,
  average the predicted NG probabilities, and use a threshold sweep on validation.
  No new architecture changes — purely diversity from stochasticity.

For iterations beyond index 2, combine elements from previous strategies or
introduce stacking if prior attempts showed strong diversity between models.

Regardless of strategy:
- The primary objective is **NG Recall >= 0.97** with **Overkill <= 0.08**.
- Do not treat lower overkill as useful if NG recall collapses, and do not treat high
  NG recall as progress if overkill remains far above 0.08. An ensemble is only useful
  if it moves the Pareto frontier toward both FN=0 and FP<=2 on the current 31 NG / 30 G
  test scale.
- If prior attempts show recall/overkill tradeoff rather than real separation, design
  the ensemble to improve G/NG separability, not just average probabilities. Prefer
  complementary evidence such as L/R absolute-difference features, calibration,
  agreement gating, or a second-stage verifier for samples near the threshold.
- For high-overkill runs, prioritize the second-stage verifier cascade over plain
  weighted averaging or max-probability threshold specialization. The verifier must
  only review samples predicted NG and must explicitly try to rescue obvious G false positives.
- You MUST set `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20` for each base model, with early stopping patience 3 epochs based on validation loss. Do NOT hardcode 5 — the line must read exactly `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20`. Keep total execution time under `config.TIMEOUT_SECONDS` (7200 s / 2 hours).
- Base each component model on the BEST_PIPELINE_SCRIPT returned by
  `load_ensemble_context_fn` — modify only the aspects required by the chosen strategy.

---
## STEP 3 — Save the ensemble strategy

Call `save_ensemble_strategy_fn` with:
- `strategy_name`         : a short kebab-case identifier (e.g. "weighted-avg-dual-backbone")
- `combination_method`    : how component outputs are merged (e.g. "weighted average of NG prob")
- `component_descriptions`: numbered list describing each model or sub-pipeline and its role,
                            as a single string

---
## STEP 4 — Write the ensemble script

Implement the chosen strategy as a single self-contained Python script.

Required properties:
- Load both _L and _R stereo images for all components that use visual input
- Load Excel labels via the split in `__DATA_SPLIT_PATH__` — the script runs as a standalone process with no ADK state access. Use: `import json; data_split = json.load(open("__DATA_SPLIT_PATH__"))`
- Seed ALL random number generators from the environment before any model or data loader
  initialization (mandatory — the evaluator injects AOI_RANDOM_SEED):
  ```python
  import os, random
  _seed = int(os.environ.get("AOI_RANDOM_SEED", os.environ.get("SEED", 42)))
  random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)
  ```
- Train all component models within the script
- Combine predictions as described in the strategy
- If implementing the second-stage verifier cascade (iteration 1), follow this exact
  structure — do NOT deviate:

  ```
  # Stage 1: high-recall detector (use best_pipeline model with LOW threshold ~0.3)
  # Run stage 1 on ALL splits so we can train the verifier on training-set stage1 outputs
  stage1_train_probs = model.predict_proba(train_samples)  # NG probability, training set
  stage1_val_probs   = model.predict_proba(val_samples)    # NG probability, val set
  stage1_test_probs  = model.predict_proba(test_samples)   # NG probability, test set
  stage1_threshold = 0.30                                  # intentionally low

  # Stage 2: verifier — trained only on samples stage1 predicts NG on the training set
  train_ng_mask = [i for i, p in enumerate(stage1_train_probs) if p >= stage1_threshold]
  verifier_X = [train_features[i] for i in train_ng_mask]  # same features used by stage1
  verifier_y = [train_labels[i]   for i in train_ng_mask]  # "G" or "NG"
  verifier.fit(verifier_X, verifier_y)

  # Cascade: only reconsider samples stage1 flagged NG; use the same feature extractor
  verifier_threshold = 0.50                                # swept on val set later
  def cascade_predict(probs, features_list):
      final_preds = []
      for s1_prob, feat in zip(probs, features_list):
          if s1_prob >= stage1_threshold:
              verifier_ng_prob = verifier.predict_proba([feat])[0]
              # Downgrade to G ONLY if verifier is confident it is G
              final_preds.append("G" if verifier_ng_prob < verifier_threshold else "NG")
          else:
              final_preds.append("G")
      return final_preds

  # Sweep (stage1_threshold, verifier_threshold) jointly on validation set
  # Priority: Stage 0 FP<=2, Stage 1 minimise miss_rate, Stage 2 minimise overkill
  ```

  Print `STAGE1_STATS: {"fp": ..., "fn": ..., "overkill": ..., "recall": ...}` before
  METRICS so stage-1 vs final improvement is visible in logs.
- Sweep threshold on validation set to maximise NG Recall subject to Overkill <= 0.08
  and the full relaxed acceptance target: miss_rate <= 0.03, ng_recall >= 0.97,
  accuracy >= 0.92. If no threshold passes, choose the lowest normalised distance
  to those targets; do not select a threshold that creates unbounded overkill.
- Include compact per-sample error analysis for FP and FN cases when the ensemble fails
  relaxed acceptance: sample identifier, true label, NG probability, threshold, and
  predicted label. Print it as `ERROR_ANALYSIS: {{...}}` after the METRICS line.
- Print EXACTLY:
  - `CALIBRATION_STATS: {{...}}` after training (G_prob_mean, G_prob_std, NG_prob_mean, NG_prob_std)
  - `THRESHOLD_CURVE: [...]` after calibration (list of {{threshold, ng_recall, overkill_rate}} dicts swept across the validation set)
  - `PREDICTIONS: [...]` after METRICS (sample_id, true_label, predicted_label, ng_probability, threshold for ALL test samples)
  - METRICS: {"accuracy": ..., "ng_recall": ..., "miss_rate": ...,
            "overkill_rate": ..., "f1": ..., "avg_latency_ms": ..., "threshold": ...,
            "ng_count": ..., "g_count": ..., "tp": ..., "tn": ..., "fp": ..., "fn": ...}
- Use train/val/test sample IDs and paths from `__DATA_SPLIT_PATH__`
- Fit all preprocessing and thresholds ONLY on training/validation — no test leakage
- Keep total wall-clock time under `config.TIMEOUT_SECONDS` (7200 s / 2 hours)

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

This final response is captured verbatim as `state["ensemble_script"]` and passed
directly to ensemble_evaluator_agent.  Any non-Python text will cause execution to fail.
"""

# Inject the real data-split checkpoint path (config-driven, not a hardcoded relative
# path). Mirrors the baseline_coder placeholder pattern.
_INSTRUCTION = _INSTRUCTION.replace("__DATA_SPLIT_PATH__", str(config.CKPT_DATA_SPLIT))

# ---------------------------------------------------------------------------
# Ensemble coder agent
# output_key captures the agent's last text response as state["ensemble_script"]
# ---------------------------------------------------------------------------

ensemble_coder_agent = LlmAgent(
    name="ensemble_coder_agent",
    model=config.MODEL_PRO,
    description=(
        "Designs and implements an ensemble strategy that combines the best pipeline "
        "with complementary approaches, validates the result, and outputs the complete "
        "ensemble script as state['ensemble_script']."
    ),
    instruction=_INSTRUCTION,
    tools=[_load_context_tool, _save_strategy_tool, code_validator_tool, store_validation_cache_tool],
    output_key="ensemble_script",
    include_contents="none",
    before_model_callback=log_context_size_callback,
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
