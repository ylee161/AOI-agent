import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.guards.code_validator_agent import code_validator_tool, store_validation_cache_tool
from mle_star_agent.shared.callbacks import count_tokens_callback, rate_limit_retry_callback
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.data_split import build_data_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FunctionTools
# ---------------------------------------------------------------------------

def ensure_data_split_fn(tool_context) -> str:
    """Load data split from checkpoint if it exists, otherwise build and save it."""
    if checkpoint_exists(config.CKPT_DATA_SPLIT):
        data = load_checkpoint(config.CKPT_DATA_SPLIT)
        tool_context.state["data_split"] = data
        modality = data.get("metadata", {}).get("input_modality", "stereo")
        tool_context.state["input_modality"] = modality
        stats = data.get("stats", {})
        return (
            f"Data split loaded from checkpoint: "
            f"train={stats.get('train_size')}, val={stats.get('val_size')}, test={stats.get('test_size')} "
            f"(NG={stats.get('ng_count')}, G={stats.get('g_count')}, total={stats.get('total')})"
            f" | input_modality={modality}"
        )

    logger.info("Building data split from %d lot folders...", len(config.DATASET_FOLDERS))
    data = build_data_split(config.DATASET_FOLDERS)
    save_checkpoint(config.CKPT_DATA_SPLIT, data)
    tool_context.state["data_split"] = data
    modality = data["metadata"]["input_modality"]
    tool_context.state["input_modality"] = modality
    stats = data.get("stats", {})
    return (
        f"Data split created and saved: "
        f"train={stats.get('train_size')}, val={stats.get('val_size')}, test={stats.get('test_size')} "
        f"(NG={stats.get('ng_count')}, G={stats.get('g_count')}, total={stats.get('total')})"
        f" | input_modality={modality}"
    )


def load_candidate_scripts_fn(tool_context) -> str:
    """Load candidate scripts from checkpoint if it exists. Returns status string."""
    if checkpoint_exists(config.CKPT_CANDIDATE_SCRIPTS):
        data = load_checkpoint(config.CKPT_CANDIDATE_SCRIPTS)
        scripts = data.get("scripts", [])
        names = [s.get("name") for s in scripts]
        if len(scripts) >= 3:
            # Only update state when fully complete to avoid stale partial state.
            tool_context.state["candidate_scripts"] = scripts
            return f"CHECKPOINT_FOUND: loaded {len(scripts)} candidate script(s): {names} — skip generation."
        else:
            return (
                f"PARTIAL_CHECKPOINT: {len(scripts)}/3 scripts already saved: {names}. "
                f"Generate and save only the missing scripts."
            )
    return "CHECKPOINT_NOT_FOUND: generate all 3 candidate scripts now."


def append_candidate_script_fn(tool_context, name: str, script: str, architecture: str) -> str:
    """Save a single validated script immediately, appending to the checkpoint."""
    if not script:
        return "ERROR: script is empty."
    existing = []
    if checkpoint_exists(config.CKPT_CANDIDATE_SCRIPTS):
        data = load_checkpoint(config.CKPT_CANDIDATE_SCRIPTS)
        existing = data.get("scripts", [])
    existing_names = [s.get("name") for s in existing]
    if name in existing_names:
        return f"Script '{name}' already saved — skipping duplicate."
    existing.append({"name": name, "script": script, "architecture": architecture})
    save_checkpoint(config.CKPT_CANDIDATE_SCRIPTS, {"scripts": existing})
    tool_context.state["candidate_scripts"] = existing
    return f"Saved script '{name}' ({len(existing)}/3 scripts saved to checkpoint)."


def _modality_loading_pattern(modality: str) -> str:
    """Return the data-loading instruction string for the given input modality."""
    if modality == "stereo":
        return (
            "Load img_l and img_r (absolute paths from split). Compute "
            "diff = abs(img_l - img_r). Concatenate -> 9-channel input. "
            "Apply identical geometric transforms to both images."
        )
    return (
        "Load img (absolute path from split). Standard 3-channel input. "
        "Standard torchvision transforms, no paired-sync requirement."
    )


def format_candidate_block(candidates, modality: str) -> str:
    """Format retrieved candidates + the modality-aware data-loading pattern for the LLM.

    `candidates` is a list of dicts with keys model_name, description, example_code
    (as produced by the retriever). Pure function (no state access) so it can be reused
    by the retriever's store tool and by `load_retrieved_candidates_fn` below.
    """
    lines = ["## Retrieved candidate models (discovered by A_retriever via web search)", ""]
    for i, c in enumerate(candidates, 1):
        name = c.get("model_name", "(unnamed)")
        # Surface the optional capacity hint so the coder can prefer the
        # lowest-capacity option on this small dataset.
        param_count = (c.get("param_count") or "").strip()
        if param_count:
            name = f"{name} (~{param_count.lstrip('~')} params)" if not param_count.lower().endswith("params") else f"{name} ({param_count})"
        lines.append(f"### Candidate {i}: {name}")
        desc = (c.get("description") or "").strip()
        if desc:
            lines.append(desc)
        code = (c.get("example_code") or "").strip()
        if code:
            lines.append("Example code:")
            lines.append("```python")
            lines.append(code)
            lines.append("```")
        lines.append("")
    lines.append("## Data-loading pattern for this dataset")
    lines.append(_modality_loading_pattern(modality))
    lines.append("")
    lines.append(f"input_modality={modality} — use this to determine data loading pattern.")
    return "\n".join(lines)


def load_retrieved_candidates_fn(tool_context) -> str:
    """Return the A_retriever model menu + modality-aware data-loading pattern.

    Reads state['retrieved_candidates'] (populated by retriever_agent, which runs before
    this agent). If empty, instructs the LLM to fall back to its own judgement — never the
    old hardcoded ResNet18/EfficientNet-B0/ResNet18+SE table.
    """
    candidates = tool_context.state.get("retrieved_candidates", [])
    modality = tool_context.state.get("input_modality", "stereo")
    if candidates:
        return format_candidate_block(candidates, modality)
    return (
        "No retrieved candidates found — use your best judgement for 3 small-data PyTorch "
        "models (prefer modern pretrained backbones such as EfficientNet, ConvNeXt-Tiny, or "
        "ViT/DeiT-Tiny; do NOT default to a legacy ResNet18-only set).\n\n"
        "## Data-loading pattern for this dataset\n"
        f"{_modality_loading_pattern(modality)}\n\n"
        f"input_modality={modality} — use this to determine data loading pattern."
    )


_ensure_data_split_tool = FunctionTool(func=ensure_data_split_fn)
_load_candidate_scripts_tool = FunctionTool(func=load_candidate_scripts_fn)
_append_candidate_script_tool = FunctionTool(func=append_candidate_script_fn)
_load_retrieved_candidates_tool = FunctionTool(func=load_retrieved_candidates_fn)

# ---------------------------------------------------------------------------
# Agent instruction
# ---------------------------------------------------------------------------

_DATA_SPLIT_CHECKPOINT = str(config.CKPT_DATA_SPLIT)
_THRESHOLD_MIN = config.THRESHOLD_MIN
_THRESHOLD_MAX = config.THRESHOLD_MAX
_THRESHOLD_STEP = config.THRESHOLD_STEP

_INSTRUCTION = f"""You are the Baseline Coder Agent in the MLE-STAR AOI inspection pipeline.

Your role: perform data split setup, then generate 3 diverse candidate training scripts for binary AOI (G/NG) classification on stereo image pairs.

---
## STEP 1 — Ensure data split

Call `ensure_data_split_fn` FIRST, before anything else. This creates or loads the 70/15/15 stratified split.

---
## STEP 2 — Check for existing candidate scripts

Call `load_candidate_scripts_fn`.
- If it returns "CHECKPOINT_FOUND" (3 scripts loaded): your job is done — do NOT generate new scripts.
- If it returns "PARTIAL_CHECKPOINT": some scripts are already saved. Generate ONLY the missing ones (skip the named ones already saved).
- If it returns "CHECKPOINT_NOT_FOUND": proceed to STEP 3 and generate all 3.

---
## STEP 3 — Generate 3 candidate training scripts

FIRST call `load_retrieved_candidates_fn`. It returns the candidate model menu discovered by
A_retriever (the retriever agent ran before you and stored 4 candidates in state) plus the
data-loading pattern for THIS dataset's input modality. Base your scripts on that menu — do
not use a fixed/hardcoded model list. If the tool reports no retrieved candidates, use your
best judgement per its fallback note.

Generate exactly 3 Python scripts. Each script must be completely self-contained — it runs as a standalone process with no ADK state access.

### Data access pattern (all scripts must follow this exactly)

```python
import json
DATA_SPLIT_PATH = "{_DATA_SPLIT_CHECKPOINT}"
with open(DATA_SPLIT_PATH) as f:
    data_split = json.load(f)

train_samples = data_split["train"]   # list of sample dicts (img_l+img_r for stereo; img for mono)
val_samples   = data_split["val"]
test_samples  = data_split["test"]
```

Each sample dict has: `sample_id` (str), the image path(s) for this dataset's input modality (`img_l` + `img_r` for stereo; `img` for mono — both absolute paths), and `label` ("G" or "NG"). Use the data-loading pattern reported by `load_retrieved_candidates_fn`.

### Mandatory requirements for every script

0. **Dry-run support** (REQUIRED — validation will fail without this): read these env vars at the top of the script and honour them throughout:
   ```python
   import os
   DRY_RUN        = os.getenv("DRY_RUN") == "1"
   DRY_RUN_EPOCHS = int(os.getenv("DRY_RUN_EPOCHS", "1"))
   DRY_RUN_SAMPLES = int(os.getenv("DRY_RUN_SAMPLES", "10"))
   ```
   - When `DRY_RUN=1`: cap `train_samples`, `val_samples`, `test_samples` to `DRY_RUN_SAMPLES` each (e.g. `train_samples = train_samples[:DRY_RUN_SAMPLES]`), and set `epochs = DRY_RUN_EPOCHS`. The script must still print `METRICS:` on exit — use whatever values come out of the capped run.
   - When `DRY_RUN` is not set or `"0"`: use all samples and the full epoch count as designed.

1. **Image loading (per input modality)**: Load images using the pattern specified by `load_retrieved_candidates_fn` output (`img_l`+`img_r` for stereo; `img` for mono). For **stereo**: load BOTH `img_l` AND `img_r`, compute `img_diff = torch.abs(img_l_tensor - img_r_tensor)`, and concatenate all three (L, R, diff) to form a **9-channel input** — the difference map directly encodes defect signal (G boards have near-zero abs(L-R), NG boards show localized high-difference regions). For **mono**: load the single `img` as a standard 3-channel input.

   **9-channel first-conv initialisation (stereo path only)**: when adapting a pretrained backbone to accept a 9-channel first conv, initialise the new layer by repeating the pretrained 3-channel weights evenly across all three groups — do NOT leave channels 4–9 randomly initialised, as this creates noisy gradients that drown out the pretrained signal on small datasets:
   ```python
   new_conv = nn.Conv2d(9, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride, padding=old_conv.padding,
                        dilation=old_conv.dilation, groups=old_conv.groups,
                        padding_mode=old_conv.padding_mode,
                        bias=old_conv.bias is not None)
   with torch.no_grad():
       new_conv.weight.data = old_conv.weight.data.repeat(1, 3, 1, 1) / 3.0
       if old_conv.bias is not None:
           new_conv.bias.data = old_conv.bias.data.clone()
   ```
   If the backbone has no pretrained first conv (e.g. trained from scratch), initialise normally.
2. **Binary labels**: G → 0, NG → 1.
3. **Weighted loss**: do NOT compute pos_weight from class counts — class balance is irrelevant to the cost asymmetry. Instead derive it from the acceptance criteria already defined in your script:
   ```python
   pos_weight = torch.tensor([OVERKILL_BUDGET / MISS_BUDGET])  # ≈ 2.67 for default config
   ```
   This encodes "I can afford to false-alarm on OVERKILL_BUDGET/MISS_BUDGET × as many good samples as I can afford to miss defects." It is self-adjusting: if config tightens MISS_BUDGET on a new dataset, pos_weight increases automatically. Pass to `BCEWithLogitsLoss(pos_weight=pos_weight.to(device))`; for `CrossEntropyLoss` (multi-class path) set `weight=torch.tensor([1.0, float(pos_weight)]).to(device)` (the weight tensor MUST be moved to `device` or CUDA logits will raise a device-mismatch error).
4. **Threshold sweep**: after training, sweep threshold from 0.01 to 0.99 inclusive with step 0.01 on the VAL set. Use acceptance-distance selection — never pick a threshold that minimises miss_rate while ignoring overkill:

   ```python
   MISS_BUDGET    = {config.MISS_RATE_RELAXED_MAX}
   RECALL_MIN     = {config.NG_RECALL_RELAXED_MIN}
   OVERKILL_BUDGET = {config.OVERKILL_RELAXED_MAX}
   ACC_MIN        = {config.ACCURACY_RELAXED_MIN}

   best_passing_candidate  = None   # set when any threshold satisfies all four constraints
   best_distance_candidate = None   # acceptance-distance of the best fallback so far
   best_fallback_threshold = 0.5
   best_threshold          = 0.5

   for threshold in [round(i / 100.0, 2) for i in range(1, 100)]:
       # ... compute tp/tn/fp/fn, then:
       miss_gap     = max(0.0, current_miss_rate     - MISS_BUDGET)     / max(MISS_BUDGET, 1e-9)
       recall_gap   = max(0.0, RECALL_MIN            - current_recall)   / max(RECALL_MIN, 1e-9)
       overkill_gap = max(0.0, current_overkill_rate - OVERKILL_BUDGET)  / max(OVERKILL_BUDGET, 1e-9)
       accuracy_gap = max(0.0, ACC_MIN               - current_accuracy) / max(ACC_MIN, 1e-9)
       current_distance = miss_gap + recall_gap + overkill_gap + accuracy_gap

       passes = (current_miss_rate <= MISS_BUDGET and current_recall >= RECALL_MIN
                 and current_overkill_rate <= OVERKILL_BUDGET and current_accuracy >= ACC_MIN)
       if passes:
           candidate = (current_miss_rate, current_overkill_rate, -current_accuracy, -current_recall, threshold)
           if best_passing_candidate is None or candidate < best_passing_candidate:
               best_passing_candidate = candidate
               best_threshold = threshold
       else:
           if best_distance_candidate is None or current_distance < best_distance_candidate:
               best_distance_candidate = current_distance
               best_fallback_threshold = threshold

   if best_passing_candidate is None:
       best_threshold = best_fallback_threshold   # balanced tradeoff, NOT min-miss-rate
   ```

   **Do NOT** filter out thresholds with `FP <= 2` as a hard gate — it silently falls back to a min-miss-rate policy when no threshold survives, which produces catastrophic overkill. The acceptance-distance fallback above handles the no-passing-threshold case with a balanced tradeoff.

   **Probability calibration before the sweep**: after training, fit `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on the VALIDATION scores (`X = raw_val_ng_scores`, `y = val_true_binary` with 1=NG, 0=G), then map all val/test scores through the fitted calibrator (`cal_probs = iso.transform(raw_scores)`) so every probability is on the calibrated [0,1] scale. Run the acceptance-distance sweep above on these CALIBRATED probabilities (the sweep already steps 0.01, finer than 0.05). The THRESHOLD_CURVE, the final test metrics, and the reported `threshold` MUST all be computed on the calibrated probabilities — never the raw scores. Labels are binary G/NG only (no defect sub-classes), so this is a single global calibrator and a single global threshold.
5. **METRICS output**: the last thing the script prints to stdout must be exactly:
   ```
   METRICS: {{"accuracy": ..., "ng_recall": ..., "miss_rate": ..., "overkill_rate": ..., "f1": ..., "avg_latency_ms": ..., "threshold": ..., "ng_count": ..., "g_count": ..., "tp": ..., "tn": ..., "fp": ..., "fn": ..., "roc_auc": ..., "prob_gap": ...}}
   ```
   `threshold` is a single float and MUST be the selected operating point on the CALIBRATED-probability scale (the isotonic calibrator's output), not the raw score scale.
   Compute all metrics on the **test split** using the best threshold from the val sweep.
   - `roc_auc`: `sklearn.metrics.roc_auc_score(y_true, ng_probs)` on the test set (y_true=1 for NG, 0 for G). If only one class present in test, emit 0.0.
   - `prob_gap`: `mean(ng_probs where true_label==NG) - mean(ng_probs where true_label==G)`. Positive = good separation; near 0 or negative = overlap problem.
6. **No data leakage**: normalisation stats must be computed on train only; threshold must be tuned on val only, never on test.
7. **Efficient execution**: use GPU if available (`.to(device)`). Full-run `epochs` MUST be set to exactly `20` with early stopping patience `3`. Do NOT hardcode 5 — the line must read exactly `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20`. Dry-run overrides this via `DRY_RUN_EPOCHS`.
8. **Learning-rate schedule** (MANDATORY): every script MUST use a real PyTorch learning-rate schedule instead of a fixed LR. Prefer either `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)` (SGDR) or `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)`. Instantiate the scheduler after the optimizer and call `scheduler.step()` correctly in the training loop: for `CosineAnnealingWarmRestarts`, step once per epoch (or batch with fractional epoch); for `ReduceLROnPlateau`, call `scheduler.step(val_loss)` after validation loss is computed. The validator will hard-reject schedule-less scripts.

   **Layer-wise LR (stereo 9-channel path only)**: when the model has a modified 9-channel input conv (initialised via the `/3` repeat above), split the optimizer into two parameter groups — the new stem needs a 10× higher LR than the pretrained backbone because it starts from a scaled-down initialisation:
   ```python
   stem_params     = list(new_conv.parameters())
   backbone_params = [p for p in model.parameters() if not any(p is sp for sp in stem_params)]
   optimizer = torch.optim.AdamW([
       {{'params': stem_params,     'lr': BASE_LR}},        # e.g. 1e-3
       {{'params': backbone_params, 'lr': BASE_LR / 10}},   # e.g. 1e-4
   ], weight_decay=1e-3)
   ```
   `BASE_LR` here is the script's own base learning rate (e.g. `1e-3`); define it once and reuse it. **Ordering matters**: build this optimizer ONLY after (a) `new_conv` has been assigned back into the model (e.g. `model.conv1 = new_conv`) so its params appear in `model.parameters()`, and (b) `model.to(device)` has run — `.to(device)` rebinds parameter identities, so an optimizer created before the device move holds stale references and silently trains nothing.
   For mono paths or feature-diff Siamese candidates (standard 3-channel convs), a single LR group is correct — do not apply this split there.
9. **Error handling**: wrap training loop in try/except and print a clear error message if something fails.
10. **Epoch logging** (MANDATORY): after each epoch print: `EPOCH_LOG: {{"epoch": N, "train_loss": X, "val_loss": X, "val_ng_recall": X, "val_overkill": X}}`
11. **Pre-training probe** (MANDATORY): before full training, run a cheap probe and print: `PROBE_METRICS: {{"ng_recall": X, "overkill_rate": X, "G_prob_mean": X, "NG_prob_mean": X, "should_continue": true/false, "reason": "..."}}`.
    - **Probe epoch count**: run **at least 5 probe epochs** when the first conv was modified for 9-channel input (the `/3`-repeat init needs warm-up time before its signal is reliable); run at least 3 probe epochs for standard 3-channel or feature-diff Siamese candidates. Under `DRY_RUN=1` these minimums do NOT apply — cap probe epochs at `min(DRY_RUN_EPOCHS, the-minimum-above)` so a dry run stays cheap.
    - **Abort conditions** — only set `should_continue: false` when the failure signal is sustained across ALL probe epochs (not just the last one): catastrophic overkill (overkill > 0.9 at threshold 0.5), recall collapse (ng_recall < 0.2 at threshold 0.5), OR no G/NG probability separation (`abs(NG_prob_mean - G_prob_mean) < 0.05`). The last condition catches cases where the model outputs near-identical scores for both classes regardless of which side of 0.5 they fall on.
12. **Calibration statistics** (MANDATORY): after training, compute NG probability for every val sample grouped by true label and print: `CALIBRATION_STATS: {{"G_prob_mean": X, "G_prob_std": X, "NG_prob_mean": X, "NG_prob_std": X}}`
13. **Threshold curve** (MANDATORY): during val sweep print: `THRESHOLD_CURVE: [{{"t": 0.1, "recall": X, "overkill": X, "miss_rate": X, "accuracy": X}}, ...]`
14. **Per-sample predictions** (MANDATORY): after METRICS print ALL test sample predictions: `PREDICTIONS: [{{"sample_id": "...", "true_label": "G", "predicted_label": "NG", "ng_probability": 0.82, "threshold": 0.35}}, ...]`

### small-data-safe strategy policy

The grouped train split has only about 287 samples. Prefer small-data-safe changes
in this priority order:
1. freeze or partially-freeze the pretrained backbone + small head, with
   partial-unfreeze adaptation preferred when full freeze underfits: freeze early backbone
   layers, unfreeze only the final ResNet block/layer4 or equivalent, and use a
   small regularized head;
2. weight decay + dropout;
3. AOI-safe augmentation;
4. smaller/regularized head;
5. local/patch or localized L/R difference evidence.

Calibration/threshold curves are REPORTING, not the only fix. They must be emitted
for diagnosis, but do not treat another threshold-only variant as the main modeling
answer.

De-prioritize and explicitly avoid: larger backbone; two-independent-backbone stereo;
global feature-difference-only; known failed mg7 threshold-only / threshold-acceptance
fingerprints; known failed mg8 global `|f_L-f_R|` fingerprint; full-freeze-only
strategy if predictions are flat or `prob_gap` collapses near zero. If using augmentation,
geometric transforms must be applied IDENTICALLY to L and R by sampling parameters
once and applying them to both. Do not use heavy color jitter, random erasing over
defects, random perspective, or L/R-desynchronizing affine/rotation/crop.

### Metric definitions (implement exactly)
- TP = true NG predicted NG; TN = true G predicted G; FP = true G predicted NG; FN = true NG predicted G
- accuracy = (TP+TN)/(TP+TN+FP+FN)
- ng_recall = TP/(TP+FN)    [if TP+FN==0 → 1.0]
- miss_rate = FN/(TP+FN)    [if TP+FN==0 → 0.0]
- overkill_rate = FP/(TN+FP)  [if TN+FP==0 → 0.0]
- f1 = 2*precision*recall/(precision+recall)  [if zero denominator → 0.0]
- avg_latency_ms = total inference time on test / len(test_samples) * 1000

### The 3 candidate architectures

Use the retriever's menu returned by `load_retrieved_candidates_fn` (4 candidates discovered
via web search). Pick 3 DISTINCT architectures from that menu and generate one self-contained
training script per pick, using each candidate's `example_code` as your starting point and
adapting it to this task. Do NOT hardcode a fixed model list — the menu is dynamic and may
differ per dataset. Apply the data-loading pattern that the tool reports for this dataset's
input modality (stereo → 9-channel L/R/diff; mono → standard 3-channel), and follow every
mandatory requirement above (loss weighting, threshold sweep, METRICS, diagnostics, dry-run).

### Optional candidate family — FEATURE-LEVEL shared-weight Siamese difference

You MAY generate a candidate from this family in place of one of the three above
(or as a refinement target later). It is a DIFFERENT mechanism from the 9-channel
pixel difference and you must not conflate the two:

- **9-channel pixel difference (Candidates 1–3 above):** `abs(img_l - img_r)` is
  computed on the raw IMAGES and concatenated to the channels BEFORE the encoder.
  The encoder sees a 9-channel tensor. There is no per-branch feature extraction.

- **Feature-level Siamese difference (this family):** a single SHARED encoder
  (the SAME module / weights) processes the left image → `f_L`, and the same shared
  encoder processes the right image → `f_R`. The absolute FEATURE difference
  `|f_L - f_R|` is computed on the encoder outputs, and the classifier head receives
  `concat([f_L, f_R, |f_L - f_R|])` — width == 3 × feature_dim. This keeps separate
  per-branch features AND an explicit learned difference signal.

If you generate a feature-diff candidate, you MUST:
1. Put the marker `FEATURE_DIFF_CANDIDATE = True` near the top of the script so the
   validator routes it to the structural feature-difference check.
2. Use ONE encoder instance for both branches (shared weights), e.g.:
   ```python
   FEATURE_DIFF_CANDIDATE = True  # feature-level Siamese difference (NOT 9-channel pixel diff)

   class SiameseDiffNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.encoder = models.resnet18(weights=None)   # ONE shared encoder
           feat_dim = self.encoder.fc.in_features          # 512 for resnet18
           self.encoder.fc = nn.Identity()
           self.fc_head = nn.Sequential(
               nn.Linear(feat_dim * 3, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1),
           )                                               # in_features == 3 × feat_dim

       def forward(self, img_l, img_r):
           f_l = self.encoder(img_l)                       # shared encoder on left
           f_r = self.encoder(img_r)                       # SAME encoder on right
           f_diff = torch.abs(f_l - f_r)                   # feature-level abs difference
           combined = torch.cat([f_l, f_r, f_diff], dim=1) # concat [f_L, f_R, |f_L-f_R|]
           return self.fc_head(combined).squeeze(1)
   ```
3. Keep the standard 3-channel stereo transform (this family does NOT use a 9-channel
   first conv). All other mandatory requirements (loss, threshold sweep, METRICS,
   diagnostics, seeding, dry-run) are unchanged.

This is the only family where `concat([f_L, f_R, |f_L - f_R|])` is required; the
validator's feature-difference check enforces shared weights + feature-level abs
difference + 3× concat width whenever the `FEATURE_DIFF_CANDIDATE` marker is present.

---
## STEP 4 — Validate and immediately save each script

For EACH generated script (that is not already in the checkpoint):
1. Call `code_validator_agent` with the script text.
2. If the validator returns "VALIDATED_SCRIPT:": extract the corrected script text.
   Call `store_validation_cache_fn` with that extracted script and status "VALIDATED".
   The extracted script MUST be byte-for-byte identical to what you pass to `append_candidate_script_fn`.
   Immediately call `append_candidate_script_fn` with name, extracted script, and architecture.
3. If the validator returns "VALIDATION_FAILED": call `store_validation_cache_fn` with the
   original script and status "VALIDATION_FAILED". Skip saving that script.

This saves each script as soon as it is validated, so progress is never lost on restart.

---
## Important rules
- Always call the tools in order: ensure_data_split → load_candidate_scripts → (if needed) generate → validate+save each one.
- Save immediately after each validation — do not batch them.
- Do not emit the scripts as plain text in your final response.
- Your final text response should be a short summary: how many scripts were saved and their names.
"""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

baseline_coder_agent = LlmAgent(
    name="baseline_coder_agent",
    model=config.MODEL_PRO,
    description=(
        "Ensures the 70/15/15 data split exists, then generates and validates "
        "3 diverse candidate training scripts for binary AOI G/NG classification."
    ),
    instruction=_INSTRUCTION,
    tools=[
        _ensure_data_split_tool,
        _load_candidate_scripts_tool,
        _load_retrieved_candidates_tool,
        _append_candidate_script_tool,
        code_validator_tool,
        store_validation_cache_tool,
    ],
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
