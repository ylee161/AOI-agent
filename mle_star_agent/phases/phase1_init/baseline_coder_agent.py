import ast
import logging
import re

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
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
        granularity = data.get("metadata", {}).get("label_granularity", "board")
        tool_context.state["input_modality"] = modality
        tool_context.state["label_granularity"] = granularity
        stats = data.get("stats", {})
        return (
            f"Data split loaded from checkpoint: "
            f"train={stats.get('train_size')}, val={stats.get('val_size')}, test={stats.get('test_size')} "
            f"(NG={stats.get('ng_count')}, G={stats.get('g_count')}, total={stats.get('total')})"
            f" | input_modality={modality} | label_granularity={granularity}"
        )

    granularity = getattr(config, "LABEL_GRANULARITY", "board")
    split_strategy = getattr(config, "SPLIT_STRATEGY", "grouped")
    logger.info(
        "Building data split from %d lot folders (label_granularity=%s, split_strategy=%s)...",
        len(config.DATASET_FOLDERS), granularity, split_strategy,
    )
    data = build_data_split(config.DATASET_FOLDERS, label_granularity=granularity, split_strategy=split_strategy)
    save_checkpoint(config.CKPT_DATA_SPLIT, data)
    tool_context.state["data_split"] = data
    modality = data["metadata"]["input_modality"]
    granularity = data["metadata"].get("label_granularity", "board")
    tool_context.state["input_modality"] = modality
    tool_context.state["label_granularity"] = granularity
    stats = data.get("stats", {})
    return (
        f"Data split created and saved: "
        f"train={stats.get('train_size')}, val={stats.get('val_size')}, test={stats.get('test_size')} "
        f"(NG={stats.get('ng_count')}, G={stats.get('g_count')}, total={stats.get('total')})"
        f" | input_modality={modality} | label_granularity={granularity}"
    )


def load_candidate_scripts_fn(tool_context) -> str:
    """Load candidate scripts from checkpoint if it exists. Returns status string."""
    if checkpoint_exists(config.CKPT_CANDIDATE_SCRIPTS):
        data = load_checkpoint(config.CKPT_CANDIDATE_SCRIPTS)
        scripts = data.get("scripts", [])
        excluded_terms = [t.lower() for t in config.HARD_EXCLUDED_ARCHITECTURES]
        # Also exclude architectures that failed at runtime in a prior session,
        # so the baseline coder never reloads scripts for a known-bad backbone.
        if checkpoint_exists(config.CKPT_FAILED_ARCHITECTURES):
            try:
                fa_data = load_checkpoint(config.CKPT_FAILED_ARCHITECTURES)
                for e in fa_data.get("failed", []):
                    for field in ("name", "architecture"):
                        val = (e.get(field) or "").strip().lower()
                        if val and val not in excluded_terms:
                            excluded_terms.append(val)
            except Exception:
                logger.warning("failed_architectures.json unreadable in baseline coder — runtime bans skipped.")
        valid, skipped = [], []
        for s in scripts:
            key = (s.get("name", "") + " " + s.get("architecture", "")).lower()
            if any(t in key for t in excluded_terms):
                skipped.append(s.get("name"))
            else:
                valid.append(s)
        if skipped:
            logger.info("Filtered hard-excluded scripts from checkpoint: %s", skipped)
        names = [s.get("name") for s in valid]
        if len(valid) >= 3:
            tool_context.state["candidate_scripts"] = valid
            return f"CHECKPOINT_FOUND: loaded {len(valid)} candidate script(s): {names} — skip generation."
        tool_context.state["candidate_scripts"] = valid
        return (
            f"PARTIAL_CHECKPOINT: {len(valid)}/3 valid scripts: {names} "
            f"(skipped hard-excluded: {skipped}). Generate and save only the missing scripts."
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


# ---------------------------------------------------------------------------
# Render-from-block path: the LLM authors ONLY the architecture block; the rest
# of the script is the canonical template byte-for-byte. This kills the entire
# free-write drift class (wrong device line, over-strict probe gates, broken
# training loops) that historically produced most Phase 1 failures.
# ---------------------------------------------------------------------------

_BLOCK_MARKER_RE = re.compile(r"#\s*<<<\s*ARCHITECTURE BLOCK (START|END)\s*>>>")

# Names the template assigns AFTER the block — a block-level assignment to any
# of them either crashes (model not built yet) or silently corrupts the run.
_BLOCK_FORBIDDEN_TOPLEVEL = {
    "model", "optimizer", "scheduler", "criterion", "device",
    "train_loader", "val_loader", "test_loader", "epochs",
}


def _architecture_block_problems(block: str) -> list:
    """Static checks on an LLM-authored architecture block. Empty list = OK."""
    problems = []
    if "def build_model" not in block:
        problems.append("block must define `def build_model()` returning the model on `device`")
    try:
        tree = ast.parse(block)
    except SyntaxError as exc:
        return [f"block has a syntax error: {exc}"]
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in _BLOCK_FORBIDDEN_TOPLEVEL:
                problems.append(
                    f"block must not assign `{target.id}` at top level — the template builds it "
                    "after the block; define build_model/build_optimizer/build_scheduler functions instead"
                )
    return problems


def append_candidate_block_fn(tool_context, name: str, architecture_block: str, architecture: str) -> str:
    """Render a complete candidate script from the canonical template plus this
    architecture block, syntax-check it, mark it pre-validated, and save it."""
    from mle_star_agent.guards.code_validator_agent import store_validation_cache_fn
    from mle_star_agent.shared.script_template import get_script_template

    block = _BLOCK_MARKER_RE.sub("", architecture_block or "").strip()
    if not block:
        return "BLOCK_REJECTED: architecture_block is empty."
    problems = _architecture_block_problems(block)
    if problems:
        return "BLOCK_REJECTED: " + "; ".join(problems) + ". Fix the block and call this tool again."
    if "PROBE_EPOCHS" not in block:
        block += "\n\nPROBE_EPOCHS = min(DRY_RUN_EPOCHS, 5) if DRY_RUN else 5"

    modality = tool_context.state.get("input_modality", "stereo")
    granularity = tool_context.state.get("label_granularity", "board")
    data_split_ckpt = getattr(config, "CKPT_DATA_SPLIT", config.CHECKPOINT_DIR / "data_split_grouped.json")
    script = get_script_template(
        data_split_path=str(data_split_ckpt),
        input_modality=modality,
        architecture_block=block,
        label_granularity=granularity,
    ).replace("__ARCHITECTURE_NAME__", architecture)
    try:
        compile(script, f"<candidate:{name}>", "exec")
    except SyntaxError as exc:
        return f"BLOCK_REJECTED: rendered script does not compile ({exc}). Fix the block and call this tool again."

    # The non-block body is the canonical template, so the script is pre-validated:
    # the slot evaluator's cache check will skip a redundant LLM validation pass.
    store_validation_cache_fn(tool_context, script, "VALIDATED")
    return append_candidate_script_fn(tool_context, name, script, architecture)


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
_append_candidate_block_tool = FunctionTool(func=append_candidate_block_fn)
_load_retrieved_candidates_tool = FunctionTool(func=load_retrieved_candidates_fn)

# ---------------------------------------------------------------------------
# Agent instruction
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are the Baseline Coder Agent in the MLE-STAR AOI inspection pipeline.

Your role: perform data split setup, then author 3 diverse candidate ARCHITECTURE BLOCKS for
binary AOI (G/NG) classification on stereo image pairs. You never write a full training
script — `append_candidate_block_fn` renders your block into the canonical template, which
already handles data loading, device selection (CUDA/MPS/CPU), pos_weight, the pre-training
probe, the training loop, isotonic calibration, the threshold sweep, and all diagnostic
output (METRICS / EPOCH_LOG / PROBE_METRICS / CALIBRATION_STATS / THRESHOLD_CURVE /
PREDICTIONS / ERROR_ANALYSIS).

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
## STEP 3 — Author 3 candidate ARCHITECTURE BLOCKS

FIRST call `load_retrieved_candidates_fn`. It returns the candidate model menu discovered by
A_retriever (the retriever agent ran before you and stored 4 candidates in state) plus the
data-loading pattern for THIS dataset's input modality. Pick 3 DISTINCT architectures from
that menu — do not use a fixed/hardcoded model list. If the tool reports no retrieved
candidates, use your best judgement for 3 small-data pretrained backbones.

For each pick, author ONLY the architecture block (a module-level Python fragment), then call
`append_candidate_block_fn(name, architecture_block, architecture)`. The tool renders the
full script, syntax-checks it, and saves it. If it returns "BLOCK_REJECTED: ...", fix the
reported problem and call it again with the corrected block.

### Block contract — what your fragment must define

1. `def build_model()` — builds the model, moves it to `device` (already defined by the
   template), and returns it. Pretrained weights are encouraged; downloads are fine (the
   harness pre-caches them in a separate pass).
2. `def build_optimizer(model)` — returns the optimizer.
3. `def build_scheduler(optimizer)` — returns a REAL LR schedule (e.g.
   `optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)`).
4. Optional `def build_criterion()` — defaults to `BCEWithLogitsLoss(pos_weight=...)` if omitted.
5. `PROBE_EPOCHS = min(DRY_RUN_EPOCHS, 5) if DRY_RUN else 5` (use 8 instead of 5 for
   feature-diff / ViT candidates — frozen backbones need more warm-up).
6. For ViT/Siamese candidates only: `FEATURE_DIFF_CANDIDATE = True` (see the mandatory rule below).

Names already in scope for your block: `torch`, `nn`, `optim`, `os`, `device`, `pos_weight`,
`DRY_RUN`, `DRY_RUN_EPOCHS`, `IMAGE_SIZE`. Single-logit binary output (`nn.Linear(..., 1)`).

NEVER assign `model`, `optimizer`, `scheduler`, `criterion`, `device`, `epochs`, or any
data loader at the top level of the block — the template builds those after your block, and
the tool rejects blocks that try.

### Input shape

- Default (9-channel pixel-diff path): the dataset feeds a 9-channel tensor
  `[L, R, |L-R|]`. Your `build_model()` must adapt the backbone's first conv to 9 input
  channels with the /3 repeat trick:
  ```python
  new_conv = nn.Conv2d(9, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                       stride=old_conv.stride, padding=old_conv.padding, bias=False)
  with torch.no_grad():
      new_conv.weight.copy_(old_conv.weight.repeat(1, 3, 1, 1) / 3.0)
  ```
  Do NOT leave channels 4-9 randomly initialised — random stem noise drowns the pretrained
  signal on this small dataset.
- Feature-diff path (`FEATURE_DIFF_CANDIDATE = True`): the model's `forward(img_l, img_r)`
  receives two standard 3-channel images; no first-conv surgery.

### Learning-rate policy (small-data critical — collapse risk)

The train split has only ~200-290 samples. A single lr=1e-3 over a whole pretrained
backbone destroys the pretrained features and collapses the model to constant output
(observed repeatedly on this dataset). Use differential parameter groups:

```python
def build_optimizer(model):
    head_params = [p for n, p in model.named_parameters() if 'fc' in n or 'classifier' in n or 'head' in n]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return optim.AdamW([
        {'params': head_params,     'lr': 1e-3},   # new layers: fast
        {'params': backbone_params, 'lr': 1e-4},   # pretrained: 10x slower (or freeze early stages)
    ], weight_decay=1e-3)
```

A modified 9-channel first conv counts as a NEW layer (it starts from a scaled-down
init) — give it the head LR, not the backbone LR. Partial freezing (e.g. freeze
everything up to the last block) is encouraged on this data size.

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

### MANDATORY rule — ViT/transformer models MUST use FEATURE-LEVEL Siamese difference

**If any candidate uses a Vision Transformer backbone** (ViT, DeiT, DINOv2, SigLIP,
CLIP, BEiT, or any model whose first layer is a patch embedding / linear projection
rather than a conv2d), you MUST use the feature-level Siamese difference family below
instead of the 9-channel pixel-diff input. Do NOT modify the patch embedding to accept
9 channels — this causes recall collapse (proven with DeiT-Small on this dataset).
Feed each stereo image as a standard 3-channel input through the SHARED frozen backbone,
then combine features as described below.

CNN backbones (EfficientNet, ResNet, MobileNet, ConvNeXt, etc.) use 9-channel pixel
diff as normal. ViT/transformer backbones always use feature-level Siamese diff.

### Feature-level shared-weight Siamese difference (required for ViT backbones)

It is a DIFFERENT mechanism from the 9-channel
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

If you author a feature-diff block, you MUST:
1. Put the marker `FEATURE_DIFF_CANDIDATE = True` in the block (the template's dataset
   and batch unpacking switch to the two-image path when it is True).
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
3. Keep `def build_model()` returning the Siamese module on `device`, plus
   build_optimizer/build_scheduler and `PROBE_EPOCHS = min(DRY_RUN_EPOCHS, 8) if DRY_RUN else 8`.

This is the only family where `concat([f_L, f_R, |f_L - f_R|])` is required (shared
weights + feature-level abs difference + 3x concat width).

---
## Important rules
- Tool order: ensure_data_split → load_candidate_scripts → (if needed) load_retrieved_candidates → author + save each block.
- Save each block via `append_candidate_block_fn` as soon as it is written — do not batch.
  The tool validates and persists; on "BLOCK_REJECTED" fix the block and retry (max 2 retries per candidate).
- Do not emit blocks or scripts as plain text in your final response.
- Your final text response should be a short summary: how many candidates were saved and their names.
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
        _append_candidate_block_tool,
    ],
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
